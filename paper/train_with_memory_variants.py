
import torch # type: ignore
import numpy as np
import absl.flags
import absl.app
import os
import yaml
import utils.utils as utils
import time
import pickle
import random
from typing import Dict, List, Tuple


# user flags
absl.flags.DEFINE_string("modality", None, "std, memory or encoder_memory")
absl.flags.DEFINE_bool("continue_train", False, "std, memory or mlp")
absl.flags.DEFINE_integer("log_interval",100,"Log interval between prints during training process")
absl.flags.DEFINE_enum(
    "memory_strategy",
    "baseline",
    ["baseline", "same_class", "different_class", "top_k", "clusters"],
    "Memory construction strategy for memory-based training.",
) # stricter 
absl.flags.mark_flag_as_required("modality")
FLAGS = absl.flags.FLAGS


def get_dataset_label(dataset:torch.utils.data.Dataset, idx:int) -> int:
    """Read class label for one item, handling common dataset layouts."""
    
    # helper
    def unwrap_dataset_index(dataset: torch.utils.data.Dataset, idx: int) -> Tuple[torch.utils.data.Dataset, int]:
        """Resolve nested Subset indices down to the underlying base dataset."""
        if isinstance(dataset, torch.utils.data.Subset):
            return unwrap_dataset_index(dataset.dataset, dataset.indices[idx])
        return dataset, idx

    base_dataset, base_idx = unwrap_dataset_index(dataset, idx)
    if hasattr(base_dataset, "targets"):
        return int(base_dataset.targets[base_idx])
    if hasattr(base_dataset, "labels"):
        return int(base_dataset.labels[base_idx])
    _, label = dataset[idx]
    return int(label)


def build_class_index_map(memory_dataset:torch.utils.data.Dataset) -> Dict[int,List[int]]:
    """Build lookup map from class label to memory dataset indices."""
    class_to_indices: Dict[int, List[int]] = {}
    for idx in range(len(memory_dataset)):
        label = get_dataset_label(memory_dataset, idx)
        class_to_indices.setdefault(label, []).append(idx)
    return class_to_indices


def build_memory_clusters(memory_dataset:torch.utils.data.Dataset, num_clusters:int, rng:random.Random, recompute_every:int=1) -> Tuple[torch.Tensor,Dict[int,List[int]]]:
    """
    Picks random memory images as centroids (flattened pixels). Assigns each sample to nearest centroid in L2.
    Every recompute_every assignments, overwrites the modified cluster centroid with the mean of its members.
    """
    n = len(memory_dataset)
    if n == 0:
        raise RuntimeError("Cannot cluster an empty memory dataset.")
    if recompute_every < 1:
        raise ValueError(f"recompute_every must be >= 1, got {recompute_every}")
    k = min(num_clusters, n)

    # picks a `k` mem imgs at random that will act as reference for centroids  # essentially initializes centroids
    centroid_sample_indices = rng.sample(range(n), k=k)
    centroid_rows = []
    for ds_idx in centroid_sample_indices:
        img = memory_dataset[ds_idx][0].cpu()
        centroid_rows.append(img.view(-1))
    cluster_centroids = torch.stack(centroid_rows, dim=0)

    def recompute_centroid(cluster_id: int) -> None:
        """Mean of flattened members; reloads from memory_dataset."""
        member_indices = cluster_to_indices[cluster_id]
        if not member_indices:
            return
        # Future: cluster_centroids[cluster_id] = memory_flat[member_indices].mean(dim=0)
        member_flats = [memory_dataset[member_idx][0].cpu().view(-1) for member_idx in member_indices]
        cluster_centroids[cluster_id] = torch.stack(member_flats, dim=0).mean(dim=0)


    # assigns every image in mem dataset to nearest centroid
    cluster_to_indices = {c: [] for c in range(k)}
    assignment_counts = {c: 0 for c in range(k)}
    for idx in range(n):
        sample_flat = memory_dataset[idx][0].cpu().view(-1)
        dists = torch.norm(cluster_centroids - sample_flat, p=2, dim=1)
        cluster_id = int(torch.argmin(dists).item()) 
        cluster_to_indices.setdefault(cluster_id, []).append(idx)
        assignment_counts[cluster_id] += 1
        if assignment_counts[cluster_id] % recompute_every == 0:
            recompute_centroid(cluster_id)

    return cluster_centroids, cluster_to_indices


def build_memory_bank(memory_dataset:torch.utils.data.Dataset) -> Tuple[torch.Tensor,torch.Tensor]:
    """Loads memory set once: flattened rows for distance search and cached images for retrieval."""
    n = len(memory_dataset)
    if n == 0:
        raise RuntimeError("Cannot build memory bank from an empty memory dataset.")
    flat_rows = []
    image_rows = []
    for idx in range(n):
        img = memory_dataset[idx][0].cpu()
        image_rows.append(img)
        flat_rows.append(img.view(-1))
    memory_flat = torch.stack(flat_rows, dim=0)
    memory_images = torch.stack(image_rows, dim=0)
    return memory_flat, memory_images


def compute_top_k_nearest_indices(anchor_image:torch.Tensor, memory_flat:torch.Tensor, k:int) -> torch.Tensor:
    """Returns indices of k nearest memory rows to anchor_image (squared L2 on flattened pixels)."""
    anchor_flat = anchor_image.view(-1)
    sq_dists = (memory_flat - anchor_flat).pow(2).sum(dim=1)
    k_safe = min(k, memory_flat.shape[0])
    return torch.topk(sq_dists, k=k_safe, largest=False).indices


def build_memory(strategy:str, data:torch.Tensor, y:torch.Tensor, mem_loader:torch.utils.data.DataLoader, memory_dataset:torch.utils.data.Dataset, class_to_indices:Dict[int,List[int]], memory_size:int, rng:random.Random, cluster_centroids:torch.Tensor, cluster_to_indices:Dict[int,List[int]], memory_flat:torch.Tensor, memory_images:torch.Tensor) -> torch.Tensor:
    """Build one batch-level memory tensor with shape [memory_size, C, H, W]."""
    # baseline (random shuffling) 
    if strategy == "baseline":
        memory_input, _ = next(iter(mem_loader))
        return memory_input

    # top k 
    if strategy == "top_k":
        if data.numel() == 0:
            raise RuntimeError("Empty input batch encountered while building top_k memory.")
        if memory_flat.numel() == 0 or memory_images.numel() == 0:
            raise RuntimeError("top_k strategy requires precomputed memory_flat and memory_images.")
        batch_anchor = data.mean(dim=0).cpu()
        nearest_indices = compute_top_k_nearest_indices(batch_anchor, memory_flat, memory_size)
        return memory_images[nearest_indices]

    # clustering
    if strategy == "clusters":
        if data.numel() == 0:
            raise RuntimeError("Empty input batch encountered while building memory for clusters strategy.")
        if cluster_centroids.numel() == 0 or not cluster_to_indices:
            raise RuntimeError("clusters strategy requires precomputed cluster_centroids and cluster_to_indices.")
        batch_anchor = data.mean(dim=0).cpu()
        anchor_flat = batch_anchor.view(-1)
        centroid_dists = torch.norm(cluster_centroids - anchor_flat, p=2, dim=1)
        nearest_cluster = int(torch.argmin(centroid_dists).item())
        candidate_indices = cluster_to_indices[nearest_cluster]
        if not candidate_indices:
            raise RuntimeError(f"Cluster {nearest_cluster} has no assigned memory indices.")
        chosen_indices = [rng.choice(candidate_indices) for _ in range(memory_size)]
        memory_images = [memory_dataset[idx][0] for idx in chosen_indices]
        return torch.stack(memory_images, dim=0)

    # same class and different class and other (error) 
    if y.numel() == 0:
        raise RuntimeError("Empty label batch encountered while building memory.")
    anchor_label = int(y[0].item())

    if strategy == "same_class":
        candidate_indices = class_to_indices[anchor_label]
    elif strategy == "different_class":
        other_labels = [lbl for lbl in class_to_indices.keys() if lbl != anchor_label]
        if not other_labels:
            raise RuntimeError("No alternative class available for different_class strategy.")
        sampled_label = rng.choice(other_labels)
        candidate_indices = class_to_indices[sampled_label]
    else: # other
        raise ValueError(f"Unsupported memory strategy: {strategy}")

    chosen_indices = [rng.choice(candidate_indices) for _ in range(memory_size)]
    memory_images = [memory_dataset[idx][0] for idx in chosen_indices]
    return torch.stack(memory_images, dim=0)


def train_memory_model(model:torch.nn.Module, loaders:List[torch.utils.data.DataLoader], optimizer:torch.optim.Optimizer, scheduler:torch.optim.lr_scheduler._LRScheduler, loss_criterion:torch.nn.modules.loss, num_epochs:int, device:torch.device, memory_strategy:str, memory_dataset:torch.utils.data.Dataset, class_to_indices:Dict[int,List[int]], memory_size:int, rng:random.Random, cluster_centroids:torch.Tensor, cluster_to_indices:Dict[int,List[int]], memory_flat:torch.Tensor, memory_images:torch.Tensor) -> torch.nn.Module:
    """Function to train a model with a Memory Wrap layer"""
    # training process
    train_loader, mem_loader = loaders 
    model.train()

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(1, num_epochs + 1):
        for batch_idx, (data, y) in enumerate(train_loader):
            
            optimizer.zero_grad()
            # input
            data = data.to(device)
            y = y.to(device)
            memory_input = build_memory(
                strategy=memory_strategy,
                data=data,
                y=y,
                mem_loader=mem_loader,
                memory_dataset=memory_dataset,
                class_to_indices=class_to_indices,
                memory_size=memory_size,
                rng=rng,
                cluster_centroids=cluster_centroids,
                cluster_to_indices=cluster_to_indices,
                memory_flat=memory_flat,
                memory_images=memory_images,
            )
            memory_input = memory_input.to(device)
            
            # perform training step
            with torch.cuda.amp.autocast():
                outputs  = model(data,memory_input)
                loss = loss_criterion(outputs, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            #log stuff
            if batch_idx % FLAGS.log_interval == 0:
                print('Train Epoch: {} [({:.0f}%({})]\t'.format(
                epoch,
                100. * batch_idx / len(train_loader), len(train_loader.dataset)),end='\r')

        scheduler.step()# increase scheduler step for each epoch

    return model


def train_std_model(model:torch.nn.Module,train_loader:torch.utils.data.DataLoader,optimizer:torch.optim.Optimizer,scheduler:torch.optim.lr_scheduler._LRScheduler, loss_criterion:torch.nn.modules.loss, num_epochs:int, device:torch.device=torch.device('cpu'))->torch.nn.Module:
    """ Function to train standard models

    Args:
        model (torch.nn.Module): standard PyTorch model
        train_loader (torch.utils.data.DataLoader): training dataset
        optimizer (torch.optim.Optimizer): PyTorch optimizer to use to perform
            training step
        scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate
        scheduler to adaptive adjusting the learning rate during training
        loss_criterion (torch.nn.modules.loss): criterion to use to compute
            the loss
        num_epochs (int): number of epoch to train the model
        device (torch.device): device where the model is stored

    Returns:
        torch.nn.Module: the trained model
    """
    # training process
    model.train()  
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(1, num_epochs + 1):      
        for batch_idx, (data, y) in enumerate(train_loader): 
            optimizer.zero_grad() 
            # input
            data = data.to(device)
            y = y.to(device)
            
            # training step
            with torch.cuda.amp.autocast():
                outputs  = model(data)
                loss = loss_criterion(outputs, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # log stuff
            if batch_idx % FLAGS.log_interval == 0:
                print('Train Epoch: {} [({:.0f}%({})]\t'.format(
                epoch,
                100. * batch_idx / len(train_loader), len(train_loader.dataset)),end='\r')
        
        scheduler.step() # increase scheduler step for each epoch

    return model


def run_experiment(config:dict, modality:str):
    """
    Method to run an experiment. Each experiment is composed by n
    runs, defined in the config dictionary, where in each of them a new
    model is trained.
    """
    # load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:{}".format(device))

    # get dataset info
    dataset_name = config['dataset_name']
    num_classes = config[dataset_name]['num_classes']

    # training parameters
    loss_criterion = torch.nn.CrossEntropyLoss()

    # saving/loading 
    save = config['save']
    path_saving_model = 'models/{}/{}/{}/{}/'.format(dataset_name,FLAGS.modality, config['model'],config['train_examples'])
    if save and not os.path.isdir(path_saving_model): 
        os.makedirs(path_saving_model)
    
    # optimizer parameters
    learning_rate = float(config['optimizer']['learning_rate'])
    weight_decay = float(config['optimizer']['weight_decay'])
    nesterov = bool(config['optimizer']['nesterov'])
    momentum = float(config['optimizer']['momentum'])
    dict_optim = {'lr' :learning_rate, 'momentum':momentum, 'weight_decay':weight_decay, 'nesterov':nesterov}
    opt_milestones = config[dataset_name]['opt_milestones']

    run_acc = []
    initial_run = 0
    if FLAGS.continue_train:
        # load model
        print("Restarting training process\n")
        info = pickle.load( open(path_saving_model+"conf.p", "rb" ) )
        initial_run = info['run_num']
        run_acc = info['accuracies']
    for run in range(initial_run,config['runs']):
        run_time = time.time()
        utils.set_seed(run)
        model = utils.get_model(config['model'],num_classes,model_type=modality)
        model = model.to(device)
        # training parameters
        optimizer = torch.optim.SGD(model.parameters(),**dict_optim)
        if dataset_name == 'CINIC10':
             scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config[dataset_name]['num_epochs'])
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,  milestones=opt_milestones)
        # get dataset and build map for strategies
        train_loader, _, test_loader, mem_loader = utils.get_loaders(config,run)
        memory_dataset = mem_loader.dataset
        class_to_indices = build_class_index_map(memory_dataset)
        memory_size = int(config[dataset_name]['mem_examples'])
        rng = random.Random(run)

        # precompute clusters if `clusters` strategy; otherwise empty
        cluster_centroids = torch.empty(0)
        cluster_to_indices = {}
        if FLAGS.memory_strategy == "clusters":
            recompute_every = 1
            num_clusters = 10
            print(f" # CLUSTERS = {num_clusters}")
            print(f" RECOMPUTE EVERY  = {recompute_every}")
            cluster_centroids, cluster_to_indices = build_memory_clusters(
                memory_dataset,
                num_clusters=num_clusters,
                rng=rng,
                recompute_every=recompute_every,
            )
            
        # precompute memory bank for top_k (flattened vectors + cached images)
        memory_flat = torch.empty(0)
        memory_images = torch.empty(0)
        if FLAGS.memory_strategy == "top_k":
            memory_flat, memory_images = build_memory_bank(memory_dataset)

        # training process
        if modality == 'memory' or modality == 'encoder_memory':
            model = train_memory_model(
                model,
                [train_loader, mem_loader],
                optimizer,
                scheduler,
                loss_criterion,
                config[dataset_name]['num_epochs'],
                device=device,
                memory_strategy=FLAGS.memory_strategy,
                memory_dataset=memory_dataset,
                class_to_indices=class_to_indices,
                memory_size=memory_size,
                rng=rng,
                cluster_centroids=cluster_centroids,
                cluster_to_indices=cluster_to_indices,
                memory_flat=memory_flat,
                memory_images=memory_images,
            )
            train_time = time.time()

            cum_acc =  []

            # perform 5 times the validation to stabilize results (due to random selection of memory samples)
            init_eval_time = time.time()
            for _ in range(5):
                best_acc, best_loss = utils.eval_memory(model,test_loader, mem_loader,loss_criterion,device)
                cum_acc.append(best_acc)
            best_acc = np.mean(cum_acc)
            end_eval_time = time.time()

        else:
            model = train_std_model(model,train_loader,optimizer,scheduler,loss_criterion,config[dataset_name]['num_epochs'],device)
            train_time = time.time()
            init_eval_time = time.time()
            best_acc, best_loss  = utils.eval_std(model,test_loader,loss_criterion,device)
            end_eval_time = time.time()

        # stats
        run_acc.append(best_acc)

        # save
        if save and path_saving_model:
            saved_name = "{}_{}.pt".format(FLAGS.memory_strategy, run+1)
            save_path = os.path.join(path_saving_model, saved_name)
            torch.save({'model_state_dict':model.state_dict(),
            'train_examples': config['train_examples'],
            'mem_examples':  config[config['dataset_name']]['mem_examples'],
            'model_name': config['model'],
            'num_classes': num_classes, 'modality':modality, 'dataset_name':config['dataset_name']} , save_path)
            info = {'run_num':run+1,'accuracies':run_acc}
            pickle.dump( info, open( path_saving_model+"conf.p", "wb" ) )

        # log
        print("Run:{} | Best Loss:{:.4f} | Accuracy {:.2f} | Mean Accuracy:{:.2f} | Std Dev Accuracy:{:.2f}\tT:{:.2f}min\tE:{:.2f}"
              .format(run+1, best_loss, best_acc, np.mean(run_acc), np.std(run_acc), (train_time-run_time)/60, (end_eval_time-init_eval_time)/60))



def main(argv):
    # load config
    config_file = open(r'config/train.yaml')
    config = yaml.safe_load(config_file)

    # run experiment
    print("Model:{}\nSizeTrain:{}\n".format(config['model'], config['train_examples']))
    run_experiment(config, FLAGS.modality)


if __name__ == '__main__':
  absl.app.run(main)
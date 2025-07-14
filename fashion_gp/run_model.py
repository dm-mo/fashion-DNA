# %%
import itertools
import json
import math
from pathlib import Path

import gpytorch
import sklearn.decomposition
import sklearn.metrics
import torch
import wandb
from torch.utils.data import DataLoader, TensorDataset

import library.gp_models
from library.likelihoods import MinibatchedDirichletClassificationLikelihood, MultitaskDirichletClassificationLikelihood

# %%
# Options
config = dict(
    use_gpu=True,
    dataset_root='F://datasets',
    # dataset_name='fashion_designers_c5',
    # checkpoint_name='fashion_designers_c5_res101_augx2.bsz_16sz_224.sgd0.002',

    dataset_name ='fashion_brands',
    # dataset_name='fashion_brands_looks',
    set_img_num = 3,
    # checkpoint_name='fashion_brands_101_res101_aug.bsz_16sz_224.sgd0.002', brand101_aug
    # checkpoint_name='fashion_brands_101_res101_aug_filtered24.bsz_16sz_224.sgd0.002', #brand24_aug
    # checkpoint_name='fashion_brands_101_res101.3layer.bsz_16sz_224.sgd0.002', #brand101'
    checkpoint_name='fashion_brands_101_res101_filtered24.bsz_16sz_224.sgd0.002', #brand101'
    # checkpoint_name='brands_c5_100looks_res101_cat_514.bsz_16sz_224.sgd0.002',  # brand101

    # checkpoint_name='imagenet',
    num_inducing = [100,200,300,400,500,600,700,800], #100
    latent_dim = 10, #[5,10,15,20,25,30,35,40,45,50], #10
    model_name='DirichletGPModel', #MultitaskDirichletGPModel DirichletGPModel MeanFieldDecoupledModel MAPApproximateGP OrthDecoupledApproximateGP
    train_fc=True,
    num_epochs=50,
    learning_rate=0.01,
)


# for latent_dim in config['latent_dim']:
for num_inducing in config['num_inducing']:

    dataset_root = config['dataset_root']
    dataset_name = config['dataset_name']
    set_img_num = config['set_img_num']
    # num_inducing = config['num_inducing']
    latent_dim = config['latent_dim']
    num_epochs = config['num_epochs']
    model_name = config['model_name']
    checkpoint_name = config['checkpoint_name']
    use_gpu = config['use_gpu']
    learning_rate = config['learning_rate']
    train_fc = config['train_fc']

    assert not use_gpu or torch.cuda.is_available(), 'GPU must be available if `use_gpu` is True'

    # %%
    # Load dataset
    torch.manual_seed(0)
    device = torch.device('cuda:0' if use_gpu else 'cpu')

    embedding_path = Path('./embeddings/') / dataset_name / checkpoint_name
    dataset_path = Path(dataset_root) / dataset_name
    # train_metadata = json.loads((dataset_path / 'train_514.json').read_text())
    # test_metadata = json.loads((dataset_path / 'test_514.json').read_text())
    # val_metadata = json.loads((dataset_path / 'val_514.json').read_text())

    train_metadata = json.loads((dataset_path / 'train_c24.json').read_text())
    test_metadata = json.loads((dataset_path / 'test_c24.json').read_text())
    val_metadata = json.loads((dataset_path / 'val_c24.json').read_text())

    n_train = len(train_metadata)
    n_test = len(test_metadata)
    n_val = len(val_metadata)

    n_classes = 1 + max(meta['label'] for meta in itertools.chain(train_metadata, test_metadata, val_metadata))

    train_classes = torch.as_tensor([meta['label'] for meta in train_metadata], dtype=torch.int)
    test_classes = torch.as_tensor([meta['label'] for meta in test_metadata], dtype=torch.int)
    val_classes = torch.as_tensor([meta['label'] for meta in val_metadata], dtype=torch.int)

    train_embeddings = torch.load(embedding_path / 'train.pt')
    test_embeddings = torch.load(embedding_path / 'test.pt')
    val_embeddings = torch.load(embedding_path / 'val.pt')
    if set_img_num > 1:
        train_embeddings = train_embeddings.reshape(train_embeddings.shape[0],-1)
        test_embeddings = test_embeddings.reshape(test_embeddings.shape[0],-1)
        val_embeddings = val_embeddings.reshape(val_embeddings.shape[0],-1)
    latent_dim_res = train_embeddings.shape[1]


    print(f'classes: {n_classes}')
    print(f'training num:{n_train}')
    print(f'testing num:{n_test}')
    print(f'validation num:{n_val}')

    # %%
    # load dataseta
    batch_size = 32
    train_dataset = TensorDataset(train_embeddings, torch.arange(n_train))
    test_dataset = TensorDataset(test_embeddings, torch.arange(n_test))
    val_dataset = TensorDataset(val_embeddings, torch.arange(n_val))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    # %%
    # training process
    best_acc = 0.0
    best_auc = 0.0
    ckt_path = Path(f'./results/') / dataset_name / checkpoint_name / model_name
    ckt_path.mkdir(exist_ok=True, parents=True)
    para_str = f'inducing_{num_inducing}dim_{latent_dim}'
    save_pth = ckt_path / f'{para_str}.pt'

    # define model
    model_class = getattr(library.gp_models, model_name, None)
    if model_class is None:
        raise NotImplementedError(f'Unknown model_name: {model_name!r}')
    model = model_class(
        num_inducing=num_inducing,
        input_dim=latent_dim_res,
        num_classes=n_classes,
        latent_dim=latent_dim
    )

    # define likelihood
    if model_class.is_multitask:
        likelihood = MultitaskDirichletClassificationLikelihood(train_classes.long(), learn_additional_noise=True)
    else:
        likelihood = MinibatchedDirichletClassificationLikelihood(train_classes.long(), learn_additional_noise=True)
    train_transformed_targets = likelihood.transformed_targets

    likelihood = likelihood.to(device)
    model = model.to(device)

    # Our loss object. We're using the VariationalELBO
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=n_train).to(device)

    if not train_fc:
        for p in model.fc.parameters():
            p.requires_grad = False
        model.fc.bias.copy_(torch.zeros_like(model.fc.bias))

        pca_matrix = sklearn.decomposition.PCA(n_components=latent_dim).fit(train_embeddings).components_
        model.fc.weight.copy_(torch.from_numpy(pca_matrix))

    # %%
    wandb.init(
        project='fashion_brands',
        config=config,
    )
    # modm
    # import os
    # os.environ["WANDB_MODE"] = "offline"
    model.train()
    mll.train()
    optimizer = torch.optim.Adam(mll.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    step = 1
    for epoch in range(num_epochs):
        model.train()
        print(f'epoch: {epoch + 1}/{num_epochs}')
        for j, (x_batch, batch_index) in enumerate(train_loader):
            if j % 50 == 0:
                print(f'{j}/{math.ceil(n_train / batch_size)}')
            optimizer.zero_grad()
            if model_class.is_multitask:
                y_batch_tf = train_transformed_targets[batch_index].to(device)
            else:
                y_batch_tf = train_transformed_targets[:, batch_index].to(device)
            output = model(x_batch.to(device))

            full_loss = -mll(output, y_batch_tf, noise_idx=batch_index)
            loss = full_loss.mean(dim=0) if not model_class.is_multitask else full_loss.div(n_classes)
            with torch.no_grad():
                wandb.log({
                    'loss': loss.item(),
                    'full_loss': wandb.Histogram(full_loss.numpy(force=True)),
                    'hyper/lengthscale': wandb.Histogram(
                        model.covar_module.base_kernel.lengthscale.view(-1).numpy(force=True)),
                    'hyper/outputscale': wandb.Histogram(model.covar_module.outputscale.numpy(force=True)),
                    'hyper/second_noise': wandb.Histogram(likelihood.second_noise.view(-1).numpy(force=True)),
                    'hyper/outputscale/second_noise': wandb.Histogram(
                        (model.covar_module.outputscale / likelihood.second_noise.view(-1)).numpy(force=True)
                    ),
                    'hyper/fc_eighs': torch.linalg.eigvalsh(model.fc.weight.T @ model.fc.weight).sum().item(),
                }, step=step)
            loss.backward()
            optimizer.step()
            step += 1

        # test on each epoch
        model.eval()
        test_pred_classes = torch.empty(n_test, dtype=torch.int64)
        test_pred_probabilities = torch.empty(n_classes, n_test, dtype=torch.float32)
        with torch.no_grad():
            for i, (x_batch, batch_index) in enumerate(test_loader):
                test_probabilities = model.compute_probabilities(x_batch.to(device))
                test_pred_classes[batch_index] = torch.argmax(test_probabilities, dim=0, keepdim=False).to('cpu')
                test_pred_probabilities[:, batch_index] = test_probabilities.to('cpu')

        metrics = {
            'acc': sklearn.metrics.accuracy_score(
                test_classes.numpy(force=True), test_pred_classes.numpy(force=True)
            ),
            'auc': sklearn.metrics.roc_auc_score(
                test_classes.numpy(force=True),
                test_pred_probabilities.numpy(force=True).T,
                multi_class='ovo'
            )
        }
        wandb.log({f'test/{k}': v for k, v in metrics.items()}, step=step)
        lr_scheduler.step()

        # save checkpoint
        if metrics['acc'] > best_acc:
            best_acc = metrics['acc']
            best_auc = metrics['auc']
            print("current best results:")
            print(metrics)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'likelihood_state_dict': likelihood.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                **metrics
            }, str(save_pth))
    print("best testing results:")
    print("best acc: {}".format(best_acc))
    print("best auc: {}".format(best_auc))
    wandb.finish()

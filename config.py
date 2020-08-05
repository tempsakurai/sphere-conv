class Config(object):
    data_path = 'data/datas/MNIST'
    epochs = 100
    use_gpu = True
    CUDA_VISIBLE_DEVICES = "2"
    batch_size = 128
    test_batch_size = 128
    optimizer = 'adam'
    lr = 1e-4
    momentum = 0.9
    save_interval = 1
    log_interval = 10
    
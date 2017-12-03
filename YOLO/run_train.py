import mxnet as mx 
from Symbol.symbol import get_resnet_model
import numpy as np
from data_ulti import get_iterator
from tools.logging_metric import LogMetricsCallback, LossMetric
import time
import argparse
from mxnet.base import _as_list

import logging
import sys
root_logger = logging.getLogger()
stdout_handler = logging.StreamHandler(sys.stdout)
root_logger.addHandler(stdout_handler)
root_logger.setLevel(logging.DEBUG)

def get_metric_stat_str(eval_metric):
    names, values = eval_metric.get()
    metric_stat_str = ""
    for j in range(len(names)):
        if j == 0:
            metric_stat_str += "%s=%f " % (names[j], values[j])
        else:
            metric_stat_str += ", %s=%f " % (names[j], values[j])
    return metric_stat_str

def callback_for_metric(name_value, callback_list):
    for callback in _as_list(callback_list):
        callback(name_value)

def callback_for_checkpoint(params, epoch_end_callback):
    symbol, arg_params, aux_params = params
    for callback in _as_list(epoch_end_callback):
        callback(epoch, symbol, arg_params, aux_params)

if __name__ == "__main__":
    # cmd line example:
    #   train small dataset
    #   - running on gpu: python run_train.py --train_data_path DATA_rec/drive_small.rec --val_data_path DATA_rec/drive_small.rec --checkpoint_prefix models/drive_small_detect --checkpoint_interval 10 --batch_size 10 --epoch 200
    #   - running on cpu: python run_train.py --cpu 1 --train_data_path DATA_rec/drive_small.rec --val_data_path DATA_rec/drive_small.rec --checkpoint_prefix models/drive_small_detect --checkpoint_interval 1 --batch_size 1 --epoch 5 --lambda_noobj 0.5
    #   train mid dataset
    #   - running on gpu: python run_train.py --train_data_path DATA_rec/drive_mid.rec  --val_data_path DATA_rec/drive_mid.rec --checkpoint_prefix models/drive_mid_detect --checkpoint_interval 50 --epoch 400 --learning_rate 0.001 --batch_size 32 --lambda_noobj 0.5

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', help='train_data_path', default="DATA_rec/training.rec")
    parser.add_argument('--val_data_path', help='val_data_path', default="DATA_rec/val.rec")
    parser.add_argument('--cpu', help='use cpu if set to 1', type=int, default=0)
    parser.add_argument('--checkpoint_prefix', help='checkpoint_prefix', default="models/drive_full_detect")
    parser.add_argument('--checkpoint_interval', type=int, default=3, help='checkpoint_interval')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning_rate')
    parser.add_argument('--lambda_noobj', type=float, default=0.2, help='lambda_noobj')
    parser.add_argument('--epoch', type=int, default=10, help='epoch')
    parser.add_argument('--log_interval', type=int, default=1, help='log_interval')
    parser.add_argument('--log_to_tensorboard', type=bool, default=True, help='log_to_tensorboard')
    parser.add_argument('--do_check_point', type=bool, default=True, help='do_check_point')

    args = parser.parse_args()
    train_data_path = args.train_data_path
    val_data_path = args.val_data_path
    checkpoint_prefix = args.checkpoint_prefix
    checkpoint_interval = args.checkpoint_interval
    batch_size = args.batch_size
    lambda_noobj = args.lambda_noobj
    learning_rate = args.learning_rate
    epoch = args.epoch
    cpu = args.cpu
    log_interval = args.log_interval
    log_to_tensorboard = args.log_to_tensorboard
    do_check_point = args.do_check_point
    if cpu == 1:
        context = mx.cpu(0)
    else:
        context = mx.gpu(0)

    # get sym
    # Try different network 18, 50, 101 to find the best one
    sym = get_resnet_model('pretrained_models/resnet-34', 0, lambda_noobj)
    _, args_params, aux_params = mx.model.load_checkpoint('pretrained_models/resnet-34', 0)

    # get some input
    # change it to the data rec you create, and modify the batch_size
    train_data = get_iterator(path=train_data_path, data_shape=(3, 224, 224), label_width=7*7*9, batch_size=batch_size, shuffle=True)
    val_data = get_iterator(path=val_data_path, data_shape=(3, 224, 224), label_width=7*7*9, batch_size=batch_size)

    # allocate gpu/cpu mem to the sym
    mod = mx.mod.Module(symbol=sym, context=context)

    # setup metric
    # metric = mx.metric.create(loss_metric, allow_extra_outputs=True)
    tme = time.time()

    # setup monitor for debugging
    def norm_stat(d):
        return mx.nd.norm(d) / np.sqrt(d.size)
    mon = None #mx.mon.Monitor(10, norm_stat, pattern=".*backward*.")

    # save model
    checkpoint = mx.callback.do_checkpoint(checkpoint_prefix, checkpoint_interval)

    batch_end_callback = [
         LogMetricsCallback('logs/train-batch' + str(tme), prefix="train-batch"),
    ]

    epoch_end_callback = [
        LogMetricsCallback('logs/train-epoch' + str(tme), prefix="train-epoch"),
    ]
    val_epoch_end_callback = [
        LogMetricsCallback('logs/val-epoch' + str(tme), prefix="val-epoch"),
    ]


    # Train
    # Try different hyperparamters to get the model converged, (batch_size,
    # optimization method, training epoch, learning rate/scheduler)

    mod.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label, for_training=True)
    mod.init_params(mx.init.Xavier(magnitude=2, rnd_type='gaussian', factor_type='in'), arg_params=args_params, aux_params=aux_params, allow_missing=True, force_init=False)
    mod.init_optimizer(optimizer='rmsprop', optimizer_params={'learning_rate':learning_rate, 'lr_scheduler': mx.lr_scheduler.FactorScheduler(300000, 0.1, 0.001)})

    eval_metric = LossMetric(0.5)

    for epoch in range(epoch):
        train_data.reset()
        eval_metric.reset()
        tic = time.time()

        for i, batch in enumerate(train_data):
            btic = time.time()
            mod.forward(batch, is_train=True)
            preds = mod.get_outputs(merge_multi_context=True)
            eval_metric.update(labels=batch.label, preds=preds)
            if log_to_tensorboard:
                names, values = eval_metric.get()
                callback_for_metric(zip(names, values), batch_end_callback)
            mod.backward()
            mod.update()

            if (i + 1) % log_interval == 0:
                print('[Epoch %d Batch %d] speed: %f samples/s, training: %s' % (epoch, i, args.batch_size / (time.time() - btic), get_metric_stat_str(eval_metric)))

        print('[Epoch %d] training: %s' % (epoch, get_metric_stat_str(eval_metric)))
        print('[Epoch %d] time cost: %f' % (epoch, time.time() - tic))
        if log_to_tensorboard:
            names, values = eval_metric.get()
            callback_for_metric(zip(names, values), epoch_end_callback)
        if do_check_point:
            arg_params, aux_params = mod.get_params()
            callback_for_checkpoint([mod.symbol, arg_params, aux_params], checkpoint)

        val_data.reset()
        eval_metric.reset()

        for i, batch in enumerate(val_data):
            mod.forward(batch, is_train=False)
            preds = mod.get_outputs(merge_multi_context=True)
            eval_metric.update(labels=batch.label, preds=preds)

        if log_to_tensorboard:
            names, values = eval_metric.get()
            callback_for_metric(zip(names, values), val_epoch_end_callback)
        print('[Epoch %d] testing: %s' % (epoch, get_metric_stat_str(eval_metric)))
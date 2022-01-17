import os
from datetime import datetime

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import torch

from src import model

from src.dataset import get_imagenet, get_cifar, get_mnist


def local_approx_noise_measure(layer, real_layer_input, real_layer_output):
    approx_input = tf.keras.Input(shape=layer.input.shape[1:])
    tempmodel = tf.keras.Model(inputs=approx_input, outputs=layer(approx_input))

    # Compute prediction
    tic_scoring_local = datetime.now()
    approx_layer_output = tempmodel.predict(real_layer_input)
    toc_scoring_local = datetime.now() - tic_scoring_local

    # Measure distance
    distance = np.linalg.norm(real_layer_output - approx_layer_output)
    return distance, np.mean(np.abs(real_layer_output)), toc_scoring_local


def global_approx_noise_measure(model, input, real_model_output):
    # Compute prediction
    tic_scoring_global = datetime.now()
    approx_model_output = model.predict(input)
    toc_scoring_global = datetime.now() - tic_scoring_global

    # Measure distance
    accuracy = np.linalg.norm(real_model_output - approx_model_output)
    return accuracy / 100, toc_scoring_global


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def pytorch_evaluate(test_dataset, model, ds_scale):
    import torch, time
    criterion = torch.nn.CrossEntropyLoss().cuda()

    def validate(val_loader, model, criterion, ds_scale):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1, top5],
            prefix='Test: ')

        # switch to evaluate mode
        model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(val_loader):
                if ds_scale and i > int(ds_scale * 98):
                    break
                if True is not None:
                    images = images.cuda(non_blocking=True)
                if torch.cuda.is_available():
                    target = target.cuda(non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # if i % 10 == 0:
                #    progress.display(i)

                #    # TODO: this should also be done with the ProgressMeter
            # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            #    .format(top1=top1, top5=top5))
        return top1.avg.cpu().numpy() / 100

    def validate_list(val_loader, model_list, ds_scale):

        # switch to evaluate mode
        top_1_list = []
        for model in model_list:
            model.cuda()
            model.eval()
            top_1_list.append(AverageMeter('Acc@1', ':6.2f'))

        with torch.no_grad():
            for i, (images, target) in enumerate(val_loader):
                if ds_scale and i > int(ds_scale * images.shape[0]):
                    break
                if True is not None:
                    images = images.cuda(non_blocking=True)
                if torch.cuda.is_available():
                    target = target.cuda(non_blocking=True)

                # compute output
                output_list = []
                for i, model in enumerate(model_list):
                    output_list.append(model(images))

                # measure accuracy
                for output, top1 in zip(output_list, top_1_list):
                    acc1 = accuracy(output, target, topk=(1,))[0]
                    top1.update(acc1[0], images.size(0))

        return [top1.avg.cpu().numpy() / 100 for top1 in top_1_list]

    if isinstance(model, list):
        return validate_list(test_dataset, model, ds_scale)
    return validate(test_dataset, model, criterion, ds_scale)


def tf_evaluate(test_dataset, model, ds_scale):
    # resolve dataset
    test_dataset = test_dataset if test_dataset \
        else get_imagenet() if list(model.input.shape[1:]) == [224, 224, 3] \
        else get_cifar() if list(model.input.shape[1:]) == [32, 32, 3] \
        else get_mnist()

    def validate(test_dataset, model: tf.keras.Model, ds_scale):
        # set number of steps
        steps = int(test_dataset.cardinality().numpy() * ds_scale) if ds_scale else None
        try:
            return model.evaluate(test_dataset, steps=steps, verbose=False)[1]
        except RuntimeError:
            tic = datetime.now()
            model.compile(
                optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=["accuracy"],
            )
            return model.evaluate(test_dataset, steps=steps, verbose=False)[1]

    def validate_list(test_dataset, model_list, ds_scale):
        top_1_list = []
        for _ in model_list:
            top_1_list.append(AverageMeter('Acc@1', ':6.2f'))

        for i, (images, labels) in enumerate(test_dataset):
            if ds_scale and i > int(ds_scale * images.shape[0]):
                break

            # compute output
            output_list = []
            for i, model in enumerate(model_list):
                output_list.append(model.predict(images))

            # measure accuracy
            for output, top1 in zip(output_list, top_1_list):
                correct = np.where(output.argmax(axis=1) == labels.numpy() if
                                   len(labels.numpy().shape) <= 1 else labels.numpy().argmax())[0].size
                total = len(images)
                top1.update(correct / total, total)
        return [top1.avg for top1 in top_1_list]

    if isinstance(model, list):
        if len(model) == 1:
            return [validate(test_dataset, model[0], ds_scale)]
        return validate_list(test_dataset, model, ds_scale)
    return validate(test_dataset, model, ds_scale)


def evaluate_model_compression(input_model: model.Model, k_vect=None, n_bit=32, n_bit_approx=8, is_clustered=False,
                               is_pruned=False,
                               is_quantized=False, ratio=True, scope="layer"):
    layer_weights_count = np.array([input_model.get_layer_weights(layer, force_numpy=True).size for layer in
                                    input_model.get_layers_of_interests()])
    baseline_network_size = layer_weights_count * n_bit
    approx_network_size = np.array([1])
    if is_pruned:
        approx_network_size = []
        for layer in input_model.get_layers_of_interests():
            weights = input_model.get_layer_weights(layer, force_numpy=True)
            assert not isinstance(weights, type(None)), f"{layer}, {type(weights)}"
            size = (len(weights.shape) * 8 + 32) * np.count_nonzero(weights)
            approx_network_size.append(size)
        approx_network_size = np.array(approx_network_size)
    elif is_clustered:
        k_vect = k_vect if isinstance(k_vect, np.ndarray) else np.array(k_vect)
        if scope == "layer":
            approx_network_size = np.ceil(np.log2(k_vect)) * layer_weights_count + k_vect * (
                n_bit if not is_quantized else n_bit_approx)
        elif scope == "channel":
            approx_network_size = 0
            for layer_shape, k_i in zip([input_model.get_layer_weights(layer, force_numpy=True).shape for layer in
                                         input_model.get_layers_of_interests()],
                                        k_vect):
                approx_network_size += layer_shape[0] * (np.prod(layer_shape[1:]) * np.ceil(np.log2(k_i)) + k_i * (
                    n_bit if not is_quantized else n_bit_approx))

            approx_network_size += shapes[:, 0] * (np.prod(layer_shape[1:]) * np.ceil(np.log2(k_i)) + k_i * (
                n_bit if not is_quantized else n_bit_approx))
        elif scope == "kernel":
            raise NotImplemented

    elif is_quantized:
        approx_network_size = layer_weights_count * n_bit_approx
    return baseline_network_size.sum() / approx_network_size.sum() if ratio else approx_network_size.sum() if is_quantized or is_pruned or is_clustered else baseline_network_size.sum()


def get_intermediate_prediction(input, mob, outputs=None):
    if outputs == None:
        outputs = [[layer.input, layer.output] for layer in mob.layers[:]]
    intermediate_layer_mob = tf.keras.Model(inputs=mob.input, outputs=outputs)
    intermediate_predict = intermediate_layer_mob.predict(input)
    return intermediate_predict

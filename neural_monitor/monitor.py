from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib
import matplotlib.pyplot as plt
import threading
import queue
import numpy as np
import collections
import pickle as pkl
import os
import time
import torch as T
import torch.nn as nn
import atexit
import logging
from matplotlib import cm
from imageio import imwrite
from shutil import copyfile
from collections import namedtuple, deque
import functools
import torch.distributed
from typing import (
    List,
    Union,
    Tuple,
    Optional,
    Iterable,
    Dict,
    Any
)
import pandas as pd
from easydict import EasyDict

try:
    import visdom
except ImportError:
    visdom = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # for Pytorch earlier than 1.1.0
    from tensorboardX import SummaryWriter

from . import utils
from .utils import root_logger, log_formatter

matplotlib.use('Agg')
__all__ = ['monitor', 'logger', 'track', 'collect_tracked_variables', 'get_tracked_variables', 'hooks']
_TRACKS = collections.OrderedDict()
hooks = {}
lock = utils.ReadWriteLock()
Git = namedtuple('Git', ('branch', 'commit_id', 'commit_message', 'commit_datetime', 'commit_user', 'commit_email'))

# setup logger
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(log_formatter)
root_logger.addHandler(consoleHandler)
logger = root_logger


def track(name: str, x: Union[T.Tensor, T.nn.Module], direction: Optional[str] = None) -> Union[T.Tensor, T.nn.Module]:
    """
    An identity function that registers hooks to
    track the value and gradient of the specified tensor.

    Here is an example of how to track an intermediate output ::

        from neural_monitor import track, get_tracked_variables
        import nueralnet_pytorch as nnt

        input = ...
        conv1 = track('op', nn.Conv2d(dim, 4, 3), 'all')
        conv2 = nn.Conv2d(4, 5, 3)
        intermediate = conv1(input)
        output = track('conv2_output', conv2(intermediate), 'all')
        loss = T.sum(output ** 2)
        loss.backward(retain_graph=True)
        d_inter = T.autograd.grad(loss, intermediate, retain_graph=True)
        d_out = T.autograd.grad(loss, output)
        tracked = get_tracked_variables()

        testing.assert_allclose(tracked['conv2_output'], nnt.utils.to_numpy(output))
        testing.assert_allclose(np.stack(tracked['grad_conv2_output']), nnt.utils.to_numpy(d_out[0]))
        testing.assert_allclose(tracked['op'], nnt.utils.to_numpy(intermediate))
        for d_inter_, tracked_d_inter_ in zip(d_inter, tracked['grad_op_output']):
            testing.assert_allclose(tracked_d_inter_, nnt.utils.to_numpy(d_inter_))

    :param name:
        name of the tracked tensor.
    :param x:
        tensor or module to be tracked.
        If module, the output of the module will be tracked.
    :param direction:
        there are 4 options

        ``None``: tracks only value.

        ``'forward'``: tracks only value.

        ``'backward'``: tracks only gradient.

        ``'all'``: tracks both value and gradient.

        Default: ``None``.
    :return: `x`.
    """

    assert isinstance(name, str), 'name must be a string, got %s' % type(name)
    assert isinstance(x, (T.nn.Module, T.Tensor)), 'x must be a Torch Module or Tensor, got %s' % type(x)
    assert direction in (
        'forward', 'backward', 'all', None), 'direction must be None, \'forward\', \'backward\', or \'all\''

    if isinstance(x, T.nn.Module):
        if direction in ('forward', 'all', None):
            def _forward_hook(module, input, output):
                _TRACKS[name] = output.detach()

            hooks[name] = x.register_forward_hook(_forward_hook)

        if direction in ('backward', 'all'):
            def _backward_hook(module, grad_input, grad_output):
                _TRACKS['grad_' + name + '_output'] = tuple([grad_out.detach() for grad_out in grad_output])

            hooks['grad_' + name + '_output'] = x.register_full_backward_hook(_backward_hook)
    else:
        if direction in ('forward', 'all', None):
            _TRACKS[name] = x.detach()

        if direction in ('backward', 'all'):
            def _hook(grad):
                _TRACKS['grad_' + name] = tuple([grad_.detach() for grad_ in grad])

            hooks['grad_' + name] = x.register_hook(_hook)

    return x


def collect_tracked_variables(name=None, return_name=False):
    """
    Gets tracked variable given name.

    :param name:
        name of the tracked variable.
        can be ``str`` or``list``/``tuple`` of ``str``s.
        If ``None``, all the tracked variables will be returned.
    :param return_name:
        whether to return the names of the tracked variables.
    :return:
        the tracked variables.
    """

    assert isinstance(name, (str, list, tuple)) or name is None, 'name must either be None, a string, or a list/tuple.'
    if name is None:
        tracked = ([n for n in _TRACKS.keys()], [val for val in _TRACKS.values()]) if return_name \
            else [val for val in _TRACKS.values()]
        return tracked
    elif isinstance(name, (list, tuple)):
        tracked = (name, [_TRACKS[n] for n in name]) if return_name else [_TRACKS[n] for n in name]
        return tracked
    else:
        tracked = (name, _TRACKS[name]) if return_name else _TRACKS[name]
        return tracked


def get_tracked_variables():
    """
    Retrieves the values of tracked variables.

    :return: a dictionary containing the values of tracked variables
        associated with the given names.
    """

    name, vars = collect_tracked_variables(return_name=True)
    dict = collections.OrderedDict()
    for n, v in zip(name, vars):
        if isinstance(v, (list, tuple)):
            dict[n] = [val for val in v] if len(v) > 1 else v[0]
        else:
            dict[n] = v
    return dict


def _spawn_defaultdict_ordereddict():
    return collections.OrderedDict()


def check_path_init(f):
    @functools.wraps(f)
    def set_default_path(self, *args, **kwargs):
        if not self._initialized:
            logger.info('Working folder not initialized! Initialize working folder to default.')
            self.initialize()
            self._initialized = True

        return f(self, *args, **kwargs)

    return set_default_path


def standardize_name(f):
    @functools.wraps(f)
    def func(self, name: str, *args, **kwargs):
        if name is not None:
            name = name.replace(' ', '-')

        f(self, name, *args, **kwargs)

    return func


def distributed_collect(f):
    @functools.wraps(f)
    def func(self, name: str, value: T.Tensor, *args, **kwargs):
        if self.distributed:
            assert isinstance(value, T.Tensor), 'value must be a Tensor in distributed mode'
            tensor_list = [torch.zeros_like(value, dtype=torch.int64) for _ in range(self.world_size)]
            T.distributed.all_gather(tensor_list, value)
        return f(self, name, value, *args, **kwargs)

    return func


def distributed_flush(f):
    @functools.wraps(f)
    def func(self, *args, **kwargs):
        if self.distributed and self.rank != 0:
            return
        return f(self, *args, **kwargs)

    return func


def standardize_image(img):
    if isinstance(img, T.Tensor):
        img = utils.to_numpy(img)

    if img.dtype != 'uint8':
        img = (255.99 * img).astype('uint8')

    if len(img.shape) == 3:
        img = np.transpose(img, (2, 0, 1))[None]
    elif len(img.shape) == 2:
        img = img[None, None]

    return img


def _convert_time_human_readable(t):
    if t < 60:
        time_unit = 's'
    elif 60 <= t < 3600:
        time_unit = 'mins'
        t /= 60.
    elif 86400 > t >= 3600:
        time_unit = 'hrs'
        t /= 3600.
    else:
        time_unit = 'days'
        t /= 86400

    return t, time_unit


class Monitor:
    """
    Collects statistics and displays the results using various backends.
    The collected stats are stored in ``<root>/<model_name>/<prefix><#id>``
    where #id is automatically assigned each time a new run starts.

    Examples
    --------
    The following snippet shows how to plot smoothed training losses and
    save images from the current iteration, and then display them every 100 iterations.

    .. code-block:: python

        from neural_monitor import monitor as mon

        # Tensorboard is turned on by default
        mon.initialize(model_name='foo-model', print_freq=100, use_tensorboard=True)
        ...

        def calculate_loss(pred, gt):
            ...
            training_loss = ...
            mon.plot('training loss', loss, smooth=.99, filter_outliers=True)

        def calculate_acc(pred, gt):
            accuracy = ...
            mon.plot('training acc', accuracy, smooth=.99, filter_outliers=True)

        ...
        for epoch in mon.iter_epoch(range(n_epochs)):
            for data in mon.iter_batch(data_loader):
                pred = net(data)
                calculate_loss(pred, gt)
                calculate_acc(pred, gt)
                mon.imwrite('input images', data['images'], latest_only=True)

            mon.dump('checkpoint.pt', {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                ...
            }, method='torch', keep=5)  # keep only 5 latest checkpoints
        ...

    Attributes
    ----------
    current_folder
        path to the current run.
    writer
        an instance of Tensorboard's :class:`SummaryWriter`
        when `use_tensorboard` is set to ``True``.
    plot_folder
        path to the folder containing the collected plots.
    file_folder
        path to the folder containing the collected files.
    image_folder
        path to the folder containing the collected images.
    hist_folder
        path to the folder containing the collected histograms.
    """
    _initialized = False
    _begin_epoch_ = 'begin_epoch'
    _end_epoch_ = 'end_epoch'
    _begin_iter_ = 'begin_iter'
    _end_iter_ = 'end_iter'
    _hparams = 'hparams'
    _hparam_metrics = 'hparam-metrics'
    _log_file = 'log.pkl'

    def __init__(self):
        self.model_name = None
        self.root = None
        self.prefix = None
        self._num_iters_per_epoch = None
        self.print_freq = 1
        self.num_iters_per_epoch = None
        self.num_epochs = None
        self.use_tensorboard = None
        self.current_folder = None
        self.plot_folder = None
        self.file_folder = None
        self.image_folder = None
        self.hist_folder = None
        self.current_run = None
        self.writer = None
        self.with_git = None
        self.git = None

        self._iter = 0
        self._last_epoch = 0
        self._num_since_beginning = collections.defaultdict(_spawn_defaultdict_ordereddict)
        self._num_since_last_flush = collections.defaultdict(_spawn_defaultdict_ordereddict)
        self._hist_since_beginning = collections.defaultdict(_spawn_defaultdict_ordereddict)
        self._hist_since_last_flush = collections.defaultdict(_spawn_defaultdict_ordereddict)
        self._mat_since_last_flush = {}
        self._img_since_last_flush = collections.defaultdict(_spawn_defaultdict_ordereddict)
        self._points_since_last_flush = collections.defaultdict(_spawn_defaultdict_ordereddict)
        self._options = collections.defaultdict(_spawn_defaultdict_ordereddict)
        self._dump_files = collections.OrderedDict()
        self._schedule = {
            self._begin_epoch_: collections.defaultdict(_spawn_defaultdict_ordereddict),
            self._end_epoch_: collections.defaultdict(_spawn_defaultdict_ordereddict),
            self._begin_iter_: collections.defaultdict(_spawn_defaultdict_ordereddict),
            self._end_iter_: collections.defaultdict(_spawn_defaultdict_ordereddict)
        }
        self._init_time = None
        self._start_epoch = None
        self._io_method = {'pickle_save': self._save_pickle, 'txt_save': self._save_txt,
                           'torch_save': self._save_torch, 'pickle_load': self._load_pickle,
                           'txt_load': self._load_txt, 'torch_load': self._load_torch}

        self._q = queue.Queue()
        self._thread = threading.Thread(target=self._flush, daemon=True)
        self.rank = None
        self.distributed = None
        self.world_size = None

        # schedule to flush when the program finishes
        atexit.register(self._atexit)

    def __setattr__(self, attr, val):
        if self._initialized and attr in ('model_name', 'root', 'current_folder', 'plot_folder', 'file_folder',
                                          'image_folder', 'hist_folder', 'current_run', 'prefix'):
            raise ValueError(f'{attr} attribute must not be set after {self.__class__.__name__} is '
                             'initialized')

        super().__setattr__(attr, val)

    def initialize(
            self,
            model_name: Optional[str] = None,
            root: Optional[str] = None,
            current_folder: Optional[str] = None,
            print_freq: Optional[int] = 1,
            num_iters_per_epoch: Optional[int] = None,
            num_epochs: Optional[int] = None,
            prefix: Optional[str] = None,
            use_tensorboard: Optional[bool] = True,
            with_git: Optional[bool] = False,
            not_found_warn: bool = True
    ) -> None:
        """
        Initializes the working directory for logging.
        If the training is distributed, this initialization should be called
        after the distributed mode has been initialized.

        :param model_name:
            name of the experimented model.
            Default: ``'my-model'``.
        :param root:
            root path to store results.
            Default: ``'results'``.
        :param current_folder:
            the folder that the experiment is currently dump to.
            Note that if `current_folder` already exists,
            all the contents will be loaded.
            This option can be used for loading a trained model.
            Default: ``None``.
        :param print_freq:
            frequency of stdout.
            Default: 1.
        :param num_iters_per_epoch:
            number of iterations per epoch.
            If not provided, it will be calculated after one epoch.
            Default: ``None``.
        :param num_epochs:
            total number of epochs.
            If provided, ETA will be shown.
            Default: ``None``.
        :param prefix:
            a common prefix that is shared between folder names of different runs.
            Default: ``'run'``.
        :param use_tensorboard:
            whether to use Tensorboard.
            Default: ``True``.
        :param with_git:
            whether to retrieve some Git information.
            Should be used only when the project is initialized with Git.
            Default: ``False``.
        :param not_found_warn:
            whether to warn when some statistics are missing from saved checkpoint.
            Default: ``True``.
        :return:
            ``None``.
        """
        if self._initialized:
            logger.warning(f'Neural Monitor has already been initialized at {self.current_folder}')
            return

        self.model_name = 'my-model' if model_name is None else model_name
        self.root = root
        self.prefix = prefix
        self._num_iters_per_epoch = num_iters_per_epoch
        self.print_freq = print_freq
        self.num_iters_per_epoch = num_iters_per_epoch
        self.num_epochs = num_epochs
        self.use_tensorboard = use_tensorboard
        self.current_folder = os.path.abspath(current_folder) if current_folder is not None else None
        self.with_git = with_git
        self.distributed = T.distributed.is_initialized()
        self.rank = T.distributed.get_rank() if self.distributed else 0
        self.world_size = T.distributed.get_world_size() if self.distributed else 1
        if self.distributed and self.rank != 0:
            logging.disable()
            return

        if with_git:
            self.init_git()
        else:
            self.git = None

        if current_folder is None:
            self.root = root = 'results' if self.root is None else self.root
            path = os.path.join(root, self.model_name)
            os.makedirs(path, exist_ok=True)
            path = self._get_new_folder(path)
            self.current_folder = os.path.normpath(path)
        else:
            self.current_folder = current_folder

        if os.path.exists(self.current_folder):
            lock.acquire_read()
            self.load_state(not_found_warn=not_found_warn)
            lock.release_read()
        else:
            os.makedirs(self.current_folder, exist_ok=True)

        # make folders to store statistics
        self.plot_folder = os.path.join(self.current_folder, 'plots')
        os.makedirs(self.plot_folder, exist_ok=True)

        self.file_folder = os.path.join(self.current_folder, 'files')
        os.makedirs(self.file_folder, exist_ok=True)

        self.image_folder = os.path.join(self.current_folder, 'images')
        os.makedirs(self.image_folder, exist_ok=True)

        self.mesh_folder = os.path.join(self.current_folder, 'meshes')
        os.makedirs(self.mesh_folder, exist_ok=True)

        self.hist_folder = os.path.join(self.current_folder, 'histograms')
        os.makedirs(self.hist_folder, exist_ok=True)

        file_handler = logging.FileHandler(f'{self.file_folder}/history.log')
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)

        root_logger.info(f'Result folder: {self.current_folder}')
        self._initialized = True
        self._init_time = time.time()
        self._start_epoch = self.epoch

        if self.use_tensorboard:
            self.init_tensorboard()

        self._thread.start()

    def state_dict(self) -> dict:
        state_dict = {
            'iter': self.iter,
            'epoch': self.epoch,
            'num_iters': self.num_iters_per_epoch,
            'num': self._num_since_beginning.copy(),
            'hist': self._hist_since_beginning.copy(),
            'options': self._options.copy()
        }
        return state_dict

    def load_state_dict(self, state_dict: dict, not_found_warn: bool = True) -> None:
        try:
            self.num_stats = state_dict['num']
        except KeyError:
            if not_found_warn:
                root_logger.warning('No record found for `num`', exc_info=True)

        try:
            self.hist_stats = state_dict['hist']
        except KeyError:
            if not_found_warn:
                root_logger.warning('No record found for `hist`', exc_info=True)

        if self.num_iters_per_epoch is None:
            try:
                self.num_iters_per_epoch = state_dict['num_iters']
            except KeyError:
                if not_found_warn:
                    root_logger.warning('No record found for `num_iters`', exc_info=True)

        try:
            self.iter = state_dict['iter']
        except KeyError:
            if not_found_warn:
                root_logger.warning('No record found for `iter`', exc_info=True)

        try:
            self.epoch = state_dict['epoch']
        except KeyError:
            if self.num_iters_per_epoch:
                self.epoch = self.iter // self.num_iters_per_epoch
            else:
                if not_found_warn:
                    root_logger.warning('No record found for `epoch`', exc_info=True)

        try:
            self._options = state_dict['options']
        except KeyError:
            if not_found_warn:
                root_logger.warning('No record found for `options`', exc_info=True)

    def load_state(self, not_found_warn=False) -> None:
        self.current_run = os.path.basename(self.current_folder)

        try:
            log = self.read_log()
            self.load_state_dict(log, not_found_warn)
        except FileNotFoundError:
            if not_found_warn:
                root_logger.warning(f'`{self._log_file}` not found in `{self.file_folder}`', exc_info=True)

    def _get_new_folder(self, path):
        runs = [folder for folder in os.listdir(path) if folder.startswith(self.prefix)]
        if not runs:
            idx = 1
        else:
            indices = sorted(
                [int(r[len(self.prefix) + 1:]) if r[len(self.prefix) + 1:].isnumeric() else 0 for r in runs])
            idx = indices[-1] + 1

        self.current_run = f'{self.prefix}-{idx}'
        return os.path.join(path, self.current_run)

    def init_tensorboard(self) -> None:
        assert self._initialized, 'Working folder must be set by set_path first.'
        if self.writer is not None:
            logger.info('Tensorboard has already been initialized!')
            return

        os.makedirs(os.path.join(self.current_folder, 'tensorboard'), exist_ok=True)
        self.writer = SummaryWriter(os.path.join(self.current_folder, 'tensorboard'))
        self.use_tensorboard = True

    def init_git(self) -> None:
        if self.git is not None:
            logger.info('Git has already been integrated!')
            return

        import git

        try:
            repo = git.Repo(os.getcwd())
            head = repo.head.reference
            self.git = Git(head.name, head.commit.hexsha, head.commit.message.rstrip(), head.commit.committed_date,
                           head.commit.author.name, head.commit.author.email)
        except git.exc.InvalidGitRepositoryError:
            self.git = None

        self.with_git = True if self.git is not None else False

    @distributed_flush
    @check_path_init
    def show_git_info(self) -> None:
        import datetime

        root_logger.info('Current branch: {}'.format(self.git.branch))
        root_logger.info('Latest commit id: {}'.format(self.git.commit_id))
        root_logger.info('Latest commit message: {}'.format(self.git.commit_message))
        root_logger.info('Latest commit date: {}'.format(datetime.datetime.fromtimestamp(self.git.commit_datetime)))

    def iter_epoch(self, iterator: Iterable) -> Any:
        """
        tracks training epoch and returns the item in `iterator`.

        :param iterator:
            the epoch iterator.
            For e.g., ``range(num_epochs)``.
        :return:
            a generator over `iterator`.

        Examples
        --------

        >>> from neural_monitor import monitor as mon
        >>> mon.print_freq = 1000
        >>> num_epochs = 10
        >>> for epoch in mon.iter_epoch(range(mon.epoch, num_epochs))
        ...     # do something here

        See Also
        --------
        :meth:`~iter_batch`
        """

        if self.num_iters_per_epoch:
            self.iter = self.epoch * self.num_iters_per_epoch

        for item in iterator:
            if self.epoch > 0 and self.num_iters_per_epoch is None:
                self.num_iters_per_epoch = self.iter // self.epoch

            yield item
            self.epoch += 1

    def iter_batch(self, iterator: Iterable) -> Any:
        """
        tracks training iteration and returns the item in `iterator`.

        :param iterator:
            the batch iterator.
            For e.g., ``enumerator(loader)``.
        :return:
            a generator over `iterator`.

        Examples
        --------

        >>> from neural_monitor import monitor as mon
        >>> mon.print_freq = 1000
        >>> data_loader = ...
        >>> num_epochs = 10
        >>> for epoch in mon.iter_epoch(range(num_epochs)):
        ...     for idx, data in mon.iter_batch(enumerate(data_loader)):
        ...         # do something here

        See Also
        --------
        :meth:`~iter_epoch`
        """

        for item in iterator:
            yield item
            if self.print_freq:
                if self.iter % self.print_freq == 0:
                    self.flush()

            self.iter += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.print_freq:
            if self.iter % self.print_freq == 0:
                self.flush()
        self.iter += 1
        if self.num_iters_per_epoch:
            self.epoch = self.iter // self.num_iters_per_epoch

    @property
    def prefix(self) -> str:
        """
        returns the prefix of saved folders.

        :return:
            :attr:`~_prefix`.
        """

        return self._prefix

    @prefix.setter
    @standardize_name
    def prefix(self, p: str):
        """
        sets the prefix of the saved folder.

        :param p:
            prefix to set.
        :return:
            ``None``.
        """
        if p is None:
            from datetime import datetime
            now = datetime.now()  # current date and time
            p = now.strftime('%m.%d.%y-%H.%M.%S')

        self._prefix = p

    @property
    def model_name(self) -> str:
        """
        returns the name of the model.

        :return:
            :attr:`~_model_name`.
        """

        return self._model_name

    @model_name.setter
    @standardize_name
    def model_name(self, name: str):
        """
        sets the name of the model.

        :param name:
            name to set.
        :return:
            ``None``.
        """

        self._model_name = name

    @property
    def iter(self) -> int:
        """
        returns the current iteration.

        :return:
            :attr:`~_iter`.
        """

        return self._iter

    @iter.setter
    def iter(self, iter: int):
        """
        sets the iteration counter to a specific value.

        :param iter:
            the iteration number to set.
        :return:
            ``None``.
        """
        assert iter >= 0, 'Iteration must be non-negative'
        self._iter = int(iter)

    @property
    def epoch(self) -> int:
        """
        returns the current epoch.

        :return:
            :attr:`~_last_epoch`.
        """

        return self._last_epoch

    @epoch.setter
    def epoch(self, epoch: int):
        """
        sets the epoch for logging and keeping training status.
        Should start from 0.

        :param epoch:
            epoch number. Should start from 0.
        :return:
            ``None``.
        """

        assert epoch >= 0, 'Epoch must be non-negative'
        self._last_epoch = int(epoch)
        if self.num_iters_per_epoch:
            self.iter = self.epoch * self.num_iters_per_epoch

    @property
    def num_stats(self):
        """
        returns the collected scalar statistics from beginning.

        :return:
            :attr:`~_num_since_beginning`.
        """

        return dict(self._num_since_beginning)

    @num_stats.setter
    def num_stats(self, stats_dict: Dict):
        self._num_since_beginning.update(stats_dict)

    @num_stats.deleter
    def num_stats(self):
        self._num_since_beginning.clear()

    def clear_num_stats(self, key):
        """
        removes the collected statistics for scalar plot of the specified `key`.

        :param key:
            the name of the scalar collection.
        :return: ``None``.
        """

        self._num_since_beginning[key].clear()

    @property
    def hist_stats(self):
        """
        returns the collected tensors from beginning.

        :return:
            :attr:`~_hist_since_beginning`.
        """

        return dict(self._hist_since_beginning)

    @hist_stats.setter
    def hist_stats(self, stats_dict: Dict):
        self._hist_since_beginning.update(stats_dict)

    @hist_stats.deleter
    def hist_stats(self):
        self._hist_since_beginning.clear()

    def clear_hist_stats(self, key: Union[int, str, Tuple]):
        """
        removes the collected statistics for histogram plot of the specified `key`.

        :param key:
            the name of the histogram collection.
        :return: ``None``.
        """

        self._hist_since_beginning[key].clear()

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, options_dict: Dict):
        self._options.update(options_dict)

    @options.deleter
    def options(self):
        self._options.clear()

    def _atexit(self):
        if self._initialized:
            self.flush()
            plt.close()
            if self.writer is not None:
                self.writer.flush()
                self.writer.close()

            self._q.join()

    @distributed_flush
    @check_path_init
    def dump_rep(self, name, obj):
        """
        saves a string representation of the given object.

        :param name:
            name of the txt file containing the string representation.
        :param obj:
            object to saved as string representation.
        :return: ``None``.
        """

        with open(os.path.join(self.current_folder, name + '.txt'), 'w') as outfile:
            outfile.write(str(obj))
            outfile.close()

    @distributed_flush
    @check_path_init
    def dump_model(self, network, use_tensorboard=False, *args, **kwargs):
        """
        saves a string representation of the given neural net.

        :param network:
            neural net to be saved as string representation.
        :param use_tensorboard:
            use tensorboard to save `network`'s graph.
        :param args:
            additional arguments to Tensorboard's :meth:`SummaryWriter`
            when `use_tensorboard` is ``True``.
        :param kwargs:
            additional keyword arguments to Tensorboard's :meth:`SummaryWriter`
            when `~se_tensorboard` is ``True``.
        :return: ``None``.
        """

        assert isinstance(network, (
            nn.Module, nn.Sequential)), 'network must be an instance of Module or Sequential, got {}'.format(
            type(network))
        self.dump_rep('network.txt', network)
        if use_tensorboard:
            self.writer.add_graph(network, *args, **kwargs)

    @distributed_flush
    @check_path_init
    def backup(self, files_or_folders: Union[str, List[str]], ignores: Union[str, List[str]] = None,
               includes: Union[str, List[str]] = None):
        """
        saves a copy of the given files to :attr:`~current_folder`.
        Accepts a str or list/tuple of file or folder names.
        You can backup your codes and/or config files for later use.

        :param files_or_folders:
            files or folders to be saved.
        :param ignores:
            files or patterns to ignore.
            Default: ``None``.
        :param includes:
            files or patterns to include.
            Default: ``None``.
        :return: ``None``.
        """
        assert isinstance(files_or_folders, (str, list, tuple)), \
            'unknown type of \'files_or_folders\'. Expect list, tuple or string, got {}'.format(type(files_or_folders))

        files_or_folders = (files_or_folders,) if isinstance(files_or_folders, str) else files_or_folders
        if ignores is None:
            ignores = ()
        elif isinstance(ignores, str):
            ignores = (ignores,)

        if includes is None:
            includes = ()
        elif isinstance(includes, str):
            includes = (includes,)

        # filter ignored and included files
        import fnmatch
        to_backup = []

        def filter_files(file_tuples):
            for tup in file_tuples:
                _, f = tup
                # this is a dirty hack. fnmatch cannot match partial string
                if not any(fnmatch.fnmatch(f, p) for p in ignores) and not any(p in f for p in ignores):
                    if includes:
                        if any(fnmatch.fnmatch(f, p) for p in includes):
                            to_backup.append(tup)
                    else:
                        to_backup.append(tup)

        for path in files_or_folders:
            if os.path.isdir(path):
                path = os.path.abspath(path)
                directory = os.path.split(path)[0]
                for root, dirs, files in os.walk(path):
                    root_dir = root[len(directory):]
                    if root_dir.startswith('/'):
                        root_dir = root_dir[1:]

                    filter_files([(os.path.join(root, f), os.path.join(root_dir, f)) for f in files])
            elif os.path.isfile(path):
                filter_files([(path, path)])

        def get_backup_filename(dst):
            if not os.path.exists(dst):
                return dst

            filename = os.path.basename(dst)
            filename, ext = os.path.splitext(filename)
            i = 1
            while True:
                new_dest = os.path.join(os.path.dirname(dst), f'{filename}_{i}{ext}')
                if os.path.exists(new_dest):
                    i += 1
                    continue
                break

            return new_dest

        for tup in to_backup:
            src, dst = tup
            try:
                dst = os.path.join(self.file_folder, dst)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                dst = get_backup_filename(dst)
                copyfile(src, f'{dst}')
            except FileNotFoundError:
                root_logger.warning('No such file or directory: %s' % src)

    @distributed_collect
    @standardize_name
    def add_hparam(self, name: str, value: Union[T.Tensor, np.ndarray, float]):
        if name not in self._options[self._hparams].keys():
            if isinstance(value, (list, tuple)):  # in distributed mode
                value = value[-1]
            if isinstance(value, T.Tensor):
                value = utils.to_numpy(value)

            self._options[self._hparams][name] = value

    @distributed_collect
    @standardize_name
    def add_metric(self, name: str, value: Union[T.Tensor, np.ndarray, float]):
        if name not in self._options[self._hparam_metrics].keys():
            if isinstance(value, (list, tuple)):  # in distributed mode
                value = value[-1]
            if isinstance(value, T.Tensor):
                value = utils.to_numpy(value)

            self._options[self._hparam_metrics][name] = value

    @distributed_collect
    @standardize_name
    def plot(self,
             name: str,
             value: Union[T.Tensor, np.ndarray, float],
             smooth: Optional[float] = 0,
             filter_outliers: Optional[bool] = True,
             precision: Optional[int] = 5,
             display: Optional[bool] = True,
             **kwargs):
        """
        schedules a plot of scalar value.
        A :mod:`matplotlib` figure will be rendered and saved every :attr:`~print_freq` iterations.

        :param name:
            name of the figure to be saved. Must be unique among plots.
        :param value:
            scalar value to be plotted.
        :param smooth:
            a value between ``0`` and ``1`` to define the smoothing window size.
            See :func:`~neuralnet_pytorch.utils.numpy_utils.smooth`.
            Default: ``0``.
        :param filter_outliers:
            whether to filter out outliers in plot.
            This affects only the plot and not the raw statistics.
            Default: True.
        :param precision:
            number of digits after decimal.
            Default: ``5``.
        :param display:
            whether to print this value.
            Default: True.
        :param kwargs:
            additional options to tensorboard.
        :return: ``None``.
        """

        self._options[name]['smooth'] = smooth
        self._options[name]['filter_outliers'] = filter_outliers
        self._options[name]['precision'] = precision
        self._options[name]['display'] = display
        if isinstance(value, (list, tuple)):
            value = sum(value) / len(value)

        if isinstance(value, T.Tensor):
            value = utils.to_numpy(value)

        self._num_since_last_flush[name][self.iter] = value
        if self.writer is not None:
            prefix = kwargs.pop('prefix', 'scalar/')
            self.writer.add_scalar(prefix + name.replace(' ', '-'), value, global_step=self.iter, **kwargs)

    def plot_hparam(self):
        try:
            self.writer.add_hparams(dict(self._options[self._hparams]), dict(self._options[self._hparam_metrics]))
        except AttributeError:
            print('Tensorboard must be initialized to use this feature')
            raise

    @distributed_collect
    @standardize_name
    def plot_matrix(self, name: str, value: Union[T.Tensor, np.ndarray, float],
                    labels: Union[List[str], List[List[str]]] = None, show_values: bool = False):
        """
        plots the given matrix with colorbar and labels if provided.

        :param name:
            name of the figure to be saved. Must be unique among plots.
        :param value:
            matrix value to be plotted.
        :param labels:
            labels of each axis.
            Can be a list/tuple of strings or a nested list/tuple.
            Defaults: None.
        :param show_values:
            show values of the matrix
        :return: ``None``.
        """

        self._options[name]['labels'] = labels
        self._options[name]['show_values'] = show_values
        if isinstance(value, (list, tuple)):
            raise ValueError('Plotting a list of matrices is not supported')

        if isinstance(value, T.Tensor):
            value = utils.to_numpy(value)

        self._mat_since_last_flush[name] = np.array(value)

    @distributed_collect
    @standardize_name
    def scatter(self, name: str,
                value: Union[T.Tensor, np.ndarray, List[T.Tensor], List[np.ndarray]],
                latest_only: bool = False,
                **kwargs):
        """
        schedules a scattor plot of (a batch of) points.
        A 3D :mod:`matplotlib` figure will be rendered and saved every :attr:`~print_freq` iterations.

        :param name:
            name of the figure to be saved. Must be unique among plots.
        :param value:
            2D or 3D tensor to be plotted. The last dim should be 3.
        :param latest_only:
            whether to save only the latest statistics or keep everything from beginning.
        :param kwargs:
            additional options to tensorboard.
        :return: ``None``.
        """

        self._options[name]['latest_only'] = latest_only
        if isinstance(value, (list, tuple)):
            value = [utils.to_numpy(v[None] if len(v.shape) == 2 else v) for v in value]
        else:
            if isinstance(value, T.Tensor):
                value = utils.to_numpy(value)

            if len(value.shape) == 2:
                value = value[None]

        self._points_since_last_flush[name][self.iter] = value
        if self.writer is not None:
            if isinstance(value, list):
                for i, v in enumerate(value):
                    self.writer.add_mesh(f'{name}-i', v, global_step=self.iter, **kwargs)
            else:
                self.writer.add_mesh(name, value, global_step=self.iter, **kwargs)

    @distributed_collect
    @standardize_name
    def imwrite(self, name: str, value: Union[T.Tensor, np.ndarray], latest_only: Optional[bool] = False, **kwargs):
        """
        schedules to save images.
        The images will be rendered and saved every :attr:`~print_freq` iterations.
        There are some assumptions about input data:

        - If the input is ``'uint8'`` it is an 8-bit image.
        - If the input is ``'float32'``, its values lie between ``0`` and ``1``.
        - If the input has 3 dims, the shape is ``[h, w, 3]`` or ``[h, w, 1]``.
        - If the channel dim is different from 3 or 1, it will be considered as multiple gray images.

        :param name:
            name of the figure to be saved. Must be unique among plots.
        :param value:
            2D, 3D or 4D tensor to be plotted.
            The expected shape is ``(H, W)`` for 2D tensor, ``(H, W, C)`` for 3D tensor and
            ``(N, C, H, W)`` for 4D tensor.
            If the number of channels is other than 3 or 1, each channel is saved as
            a gray image.
        :param latest_only:
            whether to save only the latest statistics or keep everything from beginning.
        :param kwargs:
            additional options to tensorboard.
        :return: ``None``.
        """

        self._options[name]['latest_only'] = latest_only
        value = np.concatenate([standardize_image(v) for v in value], 0) \
            if isinstance(value, (list, tuple)) else standardize_image(value)  # handler for distributed training
        self._img_since_last_flush[name][self.iter] = value
        if self.writer is not None:
            prefix = kwargs.pop('prefix', 'image/')
            self.writer.add_images(prefix + name.replace(' ', '-'), value,
                                   global_step=self.iter, dataformats='NCHW')

    @distributed_collect
    @standardize_name
    def hist(self, name, value: Union[T.Tensor, np.ndarray], n_bins: int = 20, latest_only: bool = False, **kwargs):
        """
        schedules a histogram plot of (a batch of) points.
        A :mod:`matplotlib` figure will be rendered and saved every :attr:`~print_freq` iterations.

        :param name:
            name of the figure to be saved. Must be unique among plots.
        :param value:
            any-dim tensor to be histogrammed.
        :param n_bins:
            number of bins of the histogram.
        :param latest_only:
            whether to save only the latest statistics or keep everything from beginning.
        :param kwargs:
            additional options to tensorboard
        :return: ``None``.
        """

        self._options[name]['latest_only'] = latest_only
        self._options[name]['n_bins'] = n_bins
        if isinstance(value, (list, tuple)):  # in distributed training
            value = T.stack(value)
        if isinstance(value, T.Tensor):
            value = utils.to_numpy(value)

        self._hist_since_last_flush[name][self.iter] = value
        if self.writer is not None:
            prefix = kwargs.pop('prefix', 'hist/')
            self.writer.add_histogram(prefix + name.replace(' ', '-'), value, global_step=self.iter, **kwargs)

    def save_meshes(self, name, meshes=None, verts=None, faces=None, verts_uvs=None, faces_uvs=None, texture_map=None):
        try:
            from pytorch3d import io
            from pytorch3d.structures import Meshes
            from pytorch3d.renderer import TexturesUV
        except ModuleNotFoundError:
            logger.info('Pytorch3D must be installed to use this function.')
            return

        if meshes is not None:
            assert isinstance(meshes, Meshes)
            assert verts is None and faces is None and verts_uvs is None and faces_uvs is None, \
                '`meshes` and other arguments are mutually exclusive'

        if meshes is None:
            if isinstance(verts, T.Tensor):
                if len(verts.shape) == len(faces.shape) == 2:
                    filename = os.path.join(self.mesh_folder, f'{name}.obj')
                    io.save_obj(filename, verts, faces, verts_uvs=verts_uvs, faces_uvs=faces_uvs, texture_map=texture_map)
                    return

            assert len(verts) == len(faces)
            if verts_uvs is not None:
                assert len(verts_uvs) == len(verts) == len(faces_uvs) == len(texture_map)
            else:
                verts_uvs = faces_uvs = texture_map = [None] * len(verts)
        else:
            verts = meshes.verts_list()
            faces = meshes.faces_list()
            verts_uvs = [None] * len(verts)
            faces_uvs = [None] * len(verts)
            texture_map = [None] * len(verts)
            if meshes.textures is not None:
                assert isinstance(meshes.textures, TexturesUV)
                verts_uvs = meshes.textures.verts_uvs_list()
                faces_uvs = meshes.textures.faces_uvs_list()
                texture_map = meshes.textures.maps_list()

        for idx, (v, f, v_uv, f_uv, tex) in enumerate(zip(verts, faces, verts_uvs, faces_uvs, texture_map)):
            filename = os.path.join(self.mesh_folder, f'{name}-{idx}.obj')
            io.save_obj(filename, v, f, verts_uvs=v_uv, faces_uvs=f_uv, texture_map=tex)

    def _plot(self, nums, prints):
        summary = pd.DataFrame()
        fig = plt.figure()
        plt.xlabel('iteration')
        for name, val in list(nums.items()):
            smooth = self._options[name].get('smooth')
            filter_outliers = self._options[name].get('filter_outliers')
            prec = self._options[name].get('precision')
            display = self._options[name].get('display')

            # csv summary
            tmp = pd.DataFrame(val.values(), index=val.keys(), columns=[name], dtype='float32')
            summary = summary.join(tmp, how='outer')

            # plot
            self._num_since_beginning[name].update(val)
            plt.ylabel(name)
            x_vals = sorted(self._num_since_beginning[name].keys())
            y_vals = [self._num_since_beginning[name][x] for x in x_vals]
            max_, min_, med_, mean_ = np.max(y_vals), np.min(y_vals), np.median(y_vals), np.mean(y_vals)
            argmax_, argmin_ = np.argmax(y_vals), np.argmin(y_vals)
            plt.title('max: {:.8f} at iter {} min: {:.8f} at iter {} \nmedian: {:.8f} mean: {:.8f}'
                      .format(max_, x_vals[argmax_], min_, x_vals[argmin_], med_, mean_))

            x_vals, y_vals = np.array(x_vals), np.array(y_vals)
            y_vals_smoothed = utils.smooth(y_vals, smooth)[:x_vals.shape[0]] if smooth else y_vals
            plt.plot(x_vals, y_vals_smoothed)
            if filter_outliers:
                inlier_indices = ~utils.is_outlier(y_vals)
                y_vals_filtered = y_vals[inlier_indices]
                min_, max_ = np.min(y_vals_filtered), np.max(y_vals_filtered)
                interval = (.9 ** np.sign(min_) * min_, 1.1 ** np.sign(max_) * max_)
                if not (np.any(np.isnan(interval)) or np.any(np.isinf(interval))):
                    plt.ylim(interval)

            if display:
                prints.append(f"{name}\t{np.mean(np.array(list(val.values())), 0):.{prec}f}")

            fig.savefig(os.path.join(self.plot_folder, name.replace(' ', '_') + '.jpg'))
            fig.clear()
        plt.close()
        csv_file = os.path.join(self.current_folder, 'summary.csv')
        summary.sort_index()  # makes sure the rows are in chronological order
        summary.to_csv(csv_file, mode='a', header=True if not os.path.exists(csv_file) else False)

    def _plot_matrix(self, mats):
        fig = plt.figure()
        for name, val in list(mats.items()):
            ax = fig.add_subplot(111)
            im = ax.imshow(val)
            fig.colorbar(im)

            labels = self._options[name].get('labels')
            ax.set_xticks(np.arange(len(val)))
            ax.set_yticks(np.arange(len(val)))
            if labels is not None:
                if isinstance(labels[0], (list, tuple)):
                    ax.set_xticklabels(labels[0])
                    ax.set_yticklabels(labels[1])
                else:
                    ax.set_xticklabels(labels)
                    ax.set_yticklabels(labels)

                # Rotate the tick labels and set their alignment.
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

            ax.set_ylim([-.5, len(val) - .5])

            show_values = self._options[name].get('show_values')
            if show_values:
                # Loop over data dimensions and create text annotations.
                for (i, j), z in np.ndenumerate(val):
                    ax.text(j, i, z, ha='center', va='center', color='w')

            ax.set_title(name)
            fig.savefig(os.path.join(self.plot_folder, name + '-matrix.jpg'), transparent=None)

            canvas = FigureCanvas(fig)
            canvas.draw()
            width, height = fig.get_size_inches() * fig.get_dpi()
            img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            self.writer.add_image('matrix-' + name.replace(' ', '-'), img,
                                  global_step=self.iter, dataformats='HWC')
            fig.clear()
        plt.close()

    def _imwrite(self, imgs):
        for name, val in list(imgs.items()):
            latest_only = self._options[name].get('latest_only')

            for itt, val in val.items():
                if len(val.shape) == 4:
                    for num in range(val.shape[0]):
                        img = val[num]
                        if img.shape[0] in (1, 3):
                            img = np.transpose(img, (1, 2, 0))
                            if latest_only:
                                imwrite(os.path.join(self.image_folder,
                                                     name.replace(' ', '_') + '_%d.jpg' % num), img)
                            else:
                                imwrite(os.path.join(self.image_folder,
                                                     name.replace(' ', '_') + '_%d_%d.jpg' % (itt, num)), img)

                        else:
                            for ch in range(img.shape[0]):
                                img_normed = (img[ch] - np.min(img[ch])) / (np.max(img[ch]) - np.min(img[ch]))
                                # in case all image values are the same
                                img_normed[np.isnan(img_normed)] = 0
                                img_normed[np.isinf(img_normed)] = 0
                                if latest_only:
                                    imwrite(os.path.join(
                                        self.image_folder,
                                        name.replace(' ', '_') + '_%d_%d.jpg' % (num, ch)), img_normed)
                                else:
                                    imwrite(os.path.join(
                                        self.image_folder,
                                        name.replace(' ', '_') + '_%d_%d_%d.jpg' % (itt, num, ch)), img_normed)
                else:
                    raise NotImplementedError

    def _hist(self, nums):
        fig = plt.figure()
        for name, val in list(nums.items()):
            n_bins = self._options[name].get('n_bins')
            latest_only = self._options[name].get('latest_only')
            if latest_only:
                k = max(list(nums[name].keys()))
                plt.hist(np.array(val[k]).flatten(), bins='auto')
            else:
                self._hist_since_beginning[name].update(val)

                z_vals = np.array(list(self._hist_since_beginning[name].keys()))
                vals = [np.array(self._hist_since_beginning[name][i]).flatten() for i in z_vals]
                hists = [np.histogram(v, bins=n_bins) for v in vals]
                y_vals = np.array([hists[i][0] for i in range(len(hists))])
                x_vals = np.array([hists[i][1] for i in range(len(hists))])
                x_vals = (x_vals[:, :-1] + x_vals[:, 1:]) / 2.
                z_vals = np.tile(z_vals[:, None], (1, n_bins))

                ax = fig.gca(projection='3d')
                surf = ax.plot_surface(x_vals, z_vals, y_vals, cmap=cm.coolwarm, linewidth=0, antialiased=False)
                ax.view_init(45, -90)
                fig.colorbar(surf, shrink=0.5, aspect=5)
            fig.savefig(os.path.join(self.hist_folder, name.replace(' ', '_') + '_hist.jpg'))
            fig.clear()
        plt.close()

    def _scatter(self, points):
        fig = plt.figure()
        for name, vals in list(points.items()):
            latest_only = self._options[name].get('latest_only')
            for itt, val in vals.items():
                for ii, v in enumerate(val):
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(*[v[:, i] for i in range(v.shape[-1])])
                    if latest_only:
                        plt.savefig(
                            os.path.join(self.plot_folder, name.replace(' ', '_') + '_%d.jpg' % (ii + 1)))
                    else:
                        plt.savefig(os.path.join(self.plot_folder,
                                                 name.replace(' ', '_') + '_%d_%d.jpg' % (itt, ii + 1)))

                    fig.clear()
                fig.clear()
            fig.clear()
        plt.close()

    def _flush(self):
        while True:
            items = self._q.get()
            it, epoch, nums, mats, imgs, hists, points = items
            prints = []

            with plt.xkcd():
                # plot statistics
                self._plot(nums, prints)

                # plot confusion matrix
                self._plot_matrix(mats)

                # save recorded images
                self._imwrite(imgs)

                # make histograms of recorded data
                self._hist(hists)

                # scatter point set(s)
                self._scatter(points)

            lock.acquire_write()
            with open(os.path.join(self.file_folder, self._log_file), 'wb') as f:
                state_dict = self.state_dict()
                state_dict['iter'] = it
                state_dict['epoch'] = epoch
                pkl.dump(state_dict, f, pkl.HIGHEST_PROTOCOL)
                f.close()
            lock.release_write()

            epoch_perc = (it % self.num_iters_per_epoch) / self.num_iters_per_epoch if self.num_iters_per_epoch else None
            iter_show = f'Epoch {epoch + 1} Iteration {it % self.num_iters_per_epoch}/{self.num_iters_per_epoch} ' \
                        f'({epoch_perc * 100:.2f}%)' if self.num_iters_per_epoch else f'Epoch {epoch + 1} Iteration {it}'

            elapsed_time = time.time() - self._init_time
            if self.num_iters_per_epoch and self.num_epochs:
                eta = elapsed_time / (epoch - self._start_epoch + epoch_perc + 1e-8) \
                      * (self.num_epochs - (epoch + epoch_perc))
                eta, eta_unit = _convert_time_human_readable(eta)
                eta_str = f'ETA {eta:.2f}{eta_unit}'
            else:
                eta_str = f'ETA N/A'

            elapsed_time, elapsed_time_unit = _convert_time_human_readable(elapsed_time)
            elapsed_time_str = f'{elapsed_time:.2f}{elapsed_time_unit}'
            log = f'{self.current_run}\t Elapsed time {elapsed_time_str} ({eta_str})\t{iter_show}\t' + '\t'.join(prints)
            root_logger.info(log)
            self._q.task_done()

    @distributed_flush
    @check_path_init
    def flush(self):
        """
        executes all the scheduled plots.
        Do not call this if using :class:`Monitor`'s context manager mode.

        :return: ``None``.
        """

        self._q.put((self.iter, self.epoch, self._num_since_last_flush.copy(), self._mat_since_last_flush.copy(),
                     self._img_since_last_flush.copy(), self._hist_since_last_flush.copy(),
                     self._points_since_last_flush.copy()))
        self._num_since_last_flush.clear()
        self._mat_since_last_flush.clear()
        self._img_since_last_flush.clear()
        self._hist_since_last_flush.clear()
        self._points_since_last_flush.clear()

    def _version(self, file, keep):
        name, ext = os.path.splitext(file)
        versioned_filename = os.path.normpath(name + '-%d' % self.iter + ext)

        if file not in self._dump_files.keys():
            self._dump_files[file] = deque()

        if versioned_filename not in self._dump_files[file]:
            self._dump_files[file].append(versioned_filename)

        if len(self._dump_files[file]) > keep:
            oldest_file = self._dump_files[file].popleft()
            full_file = os.path.join(self.current_folder, oldest_file)
            if os.path.exists(full_file):
                os.remove(full_file)
            else:
                root_logger.warning(f'{full_file} does not exist')

        with open(os.path.join(self.current_folder, '_version.pkl'), 'wb') as f:
            pkl.dump(self._dump_files, f, pkl.HIGHEST_PROTOCOL)
        return versioned_filename

    def save_to_table(self, table_name: str, **kwargs: Any):
        """
        Write summary into a csv table.
        Adapted from https://github.com/JonasGeiping/fullbatchtraining.

        :param table_name:
            name for the summary table.
        :param kwargs:
            fields to write into the table
        """
        import csv

        # Check for file
        fname = os.path.join(self.root, f'table_{table_name}.csv')
        fieldnames = list(kwargs.keys())

        # ensure location is the first column
        location = 'location'
        fieldnames.insert(0, location)
        kwargs[location] = self.current_folder

        # Read or write header
        try:
            with open(fname, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                header = next(reader)  # noqa  # this line is testing the header
                # assert header == fieldnames[:len(header)]  # new columns are ok, but old columns need to be consistent
                # dont test, always write when in doubt to prevent erroneous table rewrites
        except Exception as e:  # noqa
            with open(fname, 'w') as f:
                writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
                writer.writeheader()

        # Add row for this experiment
        with open(fname, 'a') as f:
            writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
            writer.writerow(kwargs)

    @distributed_flush
    @check_path_init
    @standardize_name
    def dump(self, name: str, obj: Any, method: str = 'pickle', keep: int = -1, **kwargs):
        """
        saves the given object.

        :param name:
            name of the file to be saved.
        :param obj:
            object to be saved.
        :param method:
            ``str`` or ``callable``.
            If ``callable``, it should be a custom method to dump object.
            There are 3 types of ``str``.

            ``'pickle'``: use :func:`pickle.dump` to store object.

            ``'torch'``: use :func:`torch.save` to store object.

            ``'txt'``: use :func:`numpy.savetxt` to store object.

            Default: ``'pickle'``.
        :param keep:
            the number of versions of the saved file to keep.
            Default: -1 (keeps only the latest version).
        :param kwargs:
            additional keyword arguments to the underlying save function.
        :return: ``None``.
        """
        assert callable(method) or isinstance(method, str), 'method must be a string or callable'
        if isinstance(method, str):
            assert method in ('pickle', 'torch', 'txt'), 'method must be one of \'pickle\', \'torch\', or \'txt\''

        method = method if callable(method) else self._io_method[method + '_save']
        self._dump(name.replace(' ', '_'), obj, keep, method, **kwargs)

    def load(self, file: str, method: str = 'pickle', version: int = -1, **kwargs):
        """
        loads from the given file.

        :param file:
            name of the saved file without version.
        :param method:
            ``str`` or ``callable``.
            If ``callable``, it should be a custom method to load object.
            There are 3 types of ``str``.

            ``'pickle'``: use :func:`pickle.dump` to store object.

            ``'torch'``: use :func:`torch.save` to store object.

            ``'txt'``: use :func:`numpy.savetxt` to store object.

            Default: ``'pickle'``.
        :param version:
            the version of the saved file to load.
            Default: -1 (loads the latest version of the saved file).
        :param kwargs:
            additional keyword arguments to the underlying load function.
        :return: ``None``.
        """
        assert callable(method) or isinstance(method, str), 'method must be a string or callable'
        if isinstance(method, str):
            assert method in ('pickle', 'torch', 'txt'), 'method must be one of \'pickle\', \'torch\', or \'txt\''

        method = method if callable(method) else self._io_method[method + '_load']
        return self._load(file, method, version, **kwargs)

    def _dump(self, name, obj, keep, method, **kwargs):
        assert isinstance(keep, int), 'keep must be an int, got %s' % type(keep)

        if keep < 2:
            name = os.path.join(self.current_folder, name)
            method(name, obj, **kwargs)
            root_logger.info('Object dumped to %s' % name)
        else:
            normed_name = self._version(name, keep)
            normed_name = os.path.join(self.current_folder, normed_name)
            method(normed_name, obj, **kwargs)
            root_logger.info('Object dumped to %s' % normed_name)

    def _load(self, file, method, version=-1, **kwargs):
        assert isinstance(version, int), 'keep must be an int, got %s' % type(version)

        full_file = os.path.join(self.current_folder, file)
        try:
            with open(os.path.join(self.current_folder, '_version.pkl'), 'rb') as f:
                self._dump_files = pkl.load(f)

            versions = self._dump_files.get(file, [])
            if len(versions) == 0:
                try:
                    obj = method(full_file, **kwargs)
                except FileNotFoundError:
                    root_logger.warning('No file named %s found' % file)
                    return None
            else:
                if isinstance(version, int) and version <= 0:
                    if len(versions) > 0:
                        version = versions[version]
                        obj = method(os.path.join(self.current_folder, version), **kwargs)
                    else:
                        return method(full_file, **kwargs)
                else:
                    name, ext = os.path.splitext(file)
                    file_name = os.path.normpath(name + '-%d' % version + ext)
                    if file_name in versions:
                        obj = method(os.path.join(self.current_folder, file_name), **kwargs)
                    else:
                        root_logger.warning(
                            'Version %d of %s is not found in %s' % (version, file, self.current_folder))
                        return None
        except FileNotFoundError:
            try:
                obj = method(full_file, **kwargs)
            except FileNotFoundError:
                root_logger.warning('No file named %s found' % file)
                return None

        root_logger.info('Version \'%s\' loaded' % str(version))
        return obj

    def _save_pickle(self, name, obj):
        with open(name, 'wb') as f:
            pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)
            f.close()

    def _load_pickle(self, name):
        with open(name, 'rb') as f:
            obj = pkl.load(f)
            f.close()
        return obj

    def _save_txt(self, name, obj, **kwargs):
        np.savetxt(name, obj, **kwargs)

    def _load_txt(self, name, **kwargs):
        return np.loadtxt(name, **kwargs)

    def _save_torch(self, name, obj, **kwargs):
        T.save(obj, name, **kwargs)

    def _load_torch(self, name, **kwargs):
        return T.load(name, **kwargs)

    def reset(self):
        """
        factory-resets the monitor object.
        This includes clearing all the collected data,
        set the iteration and epoch counters to 0,
        and reset the timer.

        :return: ``None``.
        """

        del self.num_stats
        del self.hist_stats
        del self.options
        self._num_since_last_flush = collections.defaultdict(_spawn_defaultdict_ordereddict)
        self._mat_since_last_flush = {}
        self._img_since_last_flush = collections.defaultdict(_spawn_defaultdict_ordereddict)
        self._hist_since_last_flush = collections.defaultdict(_spawn_defaultdict_ordereddict)
        self._points_since_last_flush = collections.defaultdict(_spawn_defaultdict_ordereddict)
        self._dump_files = collections.OrderedDict()
        self._iter = 0
        self._last_epoch = 0
        self.num_iters_per_epoch = self._num_iters_per_epoch
        self._init_time = time.time()

    def read_log(self):
        """
        reads the saved log file.

        :return:
            contents of the log file.
        """

        with open(os.path.join(self.current_folder, 'files', self._log_file), 'rb') as f:
            f.seek(0)
            try:
                contents = pkl.load(f)
            except EOFError:
                contents = {}

            f.close()
        return contents

    @staticmethod
    def print_module_summary(module, inputs, max_nesting=3, skip_redundant=True):
        """
        Stolen from NVIDIA StyleGAN.

        :param module:
            a DNN module
        :param inputs:
            sample inputs to the network
        :param max_nesting:
            the maximum of submodules to be printed out
        :param skip_redundant:
            whether to skip redundant parameters or not
        :return:
        """
        assert isinstance(module, torch.nn.Module)
        assert not isinstance(module, torch.jit.ScriptModule)
        assert isinstance(inputs, (tuple, list))

        # Register hooks.
        entries = []
        nesting = [0]

        def pre_hook(_mod, _inputs):
            nesting[0] += 1

        def post_hook(mod, _inputs, outputs):
            nesting[0] -= 1
            if nesting[0] <= max_nesting:
                outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
                outputs = [t for t in outputs if isinstance(t, torch.Tensor)]
                entries.append(EasyDict(mod=mod, outputs=outputs))

        hooks = [mod.register_forward_pre_hook(pre_hook) for mod in module.modules()]
        hooks += [mod.register_forward_hook(post_hook) for mod in module.modules()]

        # Run module.
        module.eval()
        module(*inputs)
        for hook in hooks:
            hook.remove()

        # Identify unique outputs, parameters, and buffers.
        tensors_seen = set()
        for e in entries:
            e.unique_params = [t for t in e.mod.parameters() if id(t) not in tensors_seen]
            e.unique_buffers = [t for t in e.mod.buffers() if id(t) not in tensors_seen]
            e.unique_outputs = [t for t in e.outputs if id(t) not in tensors_seen]
            tensors_seen |= {id(t) for t in e.unique_params + e.unique_buffers + e.unique_outputs}

        # Filter out redundant entries.
        if skip_redundant:
            entries = [e for e in entries if len(e.unique_params) or len(e.unique_buffers) or len(e.unique_outputs)]

        # Construct table.
        rows = [[type(module).__name__, 'Layer', 'Parameters', 'Buffers', 'Output shape', 'Datatype']]
        rows += [['---'] * len(rows[0])]
        param_total = 0
        buffer_total = 0
        submodule_names = {mod: name for name, mod in module.named_modules()}
        for e in entries:
            name = '<top-level>' if e.mod is module else submodule_names[e.mod]
            classname = e.mod._get_name()
            param_size = sum(t.numel() for t in e.unique_params)
            buffer_size = sum(t.numel() for t in e.unique_buffers)
            output_shapes = [str(list(t.shape)) for t in e.outputs]
            output_dtypes = [str(t.dtype).split('.')[-1] for t in e.outputs]
            rows += [[
                name + (':0' if len(e.outputs) >= 2 else ''),
                classname,
                str(param_size) if param_size else '-',
                str(buffer_size) if buffer_size else '-',
                (output_shapes + ['-'])[0],
                (output_dtypes + ['-'])[0],
            ]]
            for idx in range(1, len(e.outputs)):
                rows += [[name + f':{idx}', classname, '-', '-', output_shapes[idx], output_dtypes[idx]]]
            param_total += param_size
            buffer_total += buffer_size
        rows += [['---'] * len(rows[0])]
        rows += [['Total', '-', str(param_total), str(buffer_total), '-', '-']]

        # Print table.
        widths = [max(len(cell) for cell in column) for column in zip(*rows)]
        summary = '\n'
        for row in rows:
            summary += '  '.join(cell + ' ' * (width - len(cell)) for cell, width in zip(row, widths))
            summary += '\n'

        logger.info(summary)


monitor = Monitor()

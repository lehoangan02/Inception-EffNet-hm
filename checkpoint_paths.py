import os


DEFAULT_SHM_DIR = '/dev/shm'


def default_save_dir():
    env_save_dir = os.environ.get('BBAV_SAVE_DIR')
    if env_save_dir:
        return env_save_dir
    return '.'


def weights_dir(args):
    save_dir = getattr(args, 'save_dir', None) or '.'
    return os.path.join(save_dir, 'weights_' + args.dataset)


def resolve_checkpoint_path(args, checkpoint):
    if os.path.isabs(checkpoint) or os.path.dirname(checkpoint) or os.path.exists(checkpoint):
        return checkpoint
    return os.path.join(weights_dir(args), checkpoint)

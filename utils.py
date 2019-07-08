import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import tf_logging as logging


def write_results(fid, i_s, model, dataset, tag):
    with open('results.txt', 'a') as f:
        line = 'Model: {} Dataset: {} Experiment: {} FID:{} IS:{}\n'.format(model.name, dataset.name, tag, fid, i_s)
        f.write(line)
    print('FID:{} IS:{}'.format(fid, i_s))


def montage_tf(imgs, num_h, num_w):
    """Makes a montage of imgs that can be used in image_summaries.

    Args:
        imgs: Tensor of images
        num_h: Number of images per column
        num_w: Number of images per row

    Returns:
        A montage of num_h*num_w images
    """
    imgs = tf.unstack(imgs)
    img_rows = [None] * num_h
    for r in range(num_h):
        img_rows[r] = tf.concat(axis=1, values=imgs[r * num_w:(r + 1) * num_w])
    montage = tf.concat(axis=0, values=img_rows)
    return tf.expand_dims(montage, 0)


def remove_missing(var_list, model_path):
    reader = pywrap_tensorflow.NewCheckpointReader(model_path)
    if isinstance(var_list, dict):
        var_dict = var_list
    else:
        var_dict = {var.op.name: var for var in var_list}
    available_vars = {}
    for var in var_dict:

        if reader.has_tensor(var):
            available_vars[var] = var_dict[var]
        else:
            logging.warning(
                'Variable %s missing in checkpoint %s', var, model_path)
    var_list = available_vars
    return var_list


def get_variables_to_train(trainable_scopes=None):
    """Returns a list of variables to train.
    Returns:
      A list of variables to train by the optimizer.
    """
    if trainable_scopes is None:
        variables_to_train = tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in trainable_scopes.split(',')]

        variables_to_train = []
        for scope in scopes:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            variables_to_train.extend(variables)

    print('Variables to train: {}'.format([v.op.name for v in variables_to_train]))

    return variables_to_train


def get_checkpoint_path(checkpoint_dir):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if not ckpt:
        print("No checkpoint in {}".format(checkpoint_dir))
        return None
    return ckpt.model_checkpoint_path


def get_all_checkpoint_paths(checkpoint_dir):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if not ckpt:
        print("No checkpoint in {}".format(checkpoint_dir))
        return None
    return ckpt.all_model_checkpoint_paths

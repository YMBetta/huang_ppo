import tensorflow as tf
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])
    tf.get_variable_scope().reuse_variables()
    print(tf.get_variable_scope().name)
    v1 = tf.get_variable("v", [1])
assert v1 == v
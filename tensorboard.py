import tensorflow as tf

with tf.name_scope('Operation_A'):
    a = tf.add(1, 2, name='a_add')
    a1 = tf.add(100, 200, name='a1_add')
    a2 = tf.multiply(a, a1, name='a2_multiply')

with tf.name_scope('Operation_B'):
    b = tf.add(3, 4, name='b_add')
    b1 = tf.add(300, 400, name='b1_add')
    b2 = tf.multiply(b, b1, name='b2_multiply')

c = tf.multiply(a2, b2, name='final_result')

*** Session oluşturalım ve summary.FileWriter() methodu ile verileri oluşturalım. run() methodu ile işlemleri başlatalım. İşlemler bittiğinde FileWriter() sınıfından ürettiğimiz objeyi close() methodu ile kapatmayı unutmayalım:

with tf.Session() as sess:
    writer = tf.summary.FileWriter('tensorboard_test', sess.graph)
    result = sess.run(c)
    writer.close()

print(result)

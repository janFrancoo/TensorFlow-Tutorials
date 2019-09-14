import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

tf.logging.set_verbosity(tf.logging.INFO)

cols = ["Number_pregnant", "Glucose_concentration", "Blood_pressure", "Triceps", "Insulin", "BMI",
        "Pedigree", "Age", "Class"]
df = pd.read_csv("pima-indians-diabetes.csv", names=cols)
print(df.head())

df[cols[:7]] = df[cols[:7]].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
print(df.head())

num_preg = tf.feature_column.numeric_column('Number_pregnant')
glu_con = tf.feature_column.numeric_column('Glucose_concentration')
blood_pre = tf.feature_column.numeric_column('Blood_pressure')
triceps = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')

df['Age'].hist(bins=20)
plt.show()

age_bucket = tf.feature_column.bucketized_column(age, boundaries=[20, 30, 40, 50, 60, 70, 80])
feat_cols = [num_preg, glu_con, blood_pre, triceps, insulin, bmi, pedigree, age_bucket]

x_data = df.drop("Class", axis=1)
y_data = df['Class']

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=.33,
                                                    random_state=101)

input_func = tf.estimator.inputs.pandas_input_fn(x=x_train, y=y_train, batch_size=10,
                                                 num_epochs=1000, shuffle=True)
model = tf.estimator.LinearClassifier(feature_columns=feat_cols, n_classes=2)
train = model.train(input_fn=input_func, steps=1000)

test_input_func = tf.estimator.inputs.pandas_input_fn(x=x_test, y=y_test, batch_size=10,
                                                      num_epochs=1, shuffle=False)
results = model.evaluate(test_input_func)
print(results)

# Credit_card_customers

In repository there is the project made during one of recruitment processes in which I participated
The task was to build a model that would predict the probability of the event that the customer would end the relationship 
with the bank and identify the factors responsible for it so as to be able to try to change the customers' decision to leave. 

# Dataset
The dataset (BankChurners.csv) can be found on the kaggle platform at the link:
https://www.kaggle.com/sakshigoyal7/credit-card-customers
Dataset contains bank customers about whom we have basic demographic, product and transaction information.

# Correlation of features with the decision 
An analysis of the impact of the features stored in the database on the customers' decision to terminate their cooperation 
with the bank was carried out. Categorical features were divided according to the representation of one-hot encoding, 
which made it possible to study the impact of each of the individual feature categories on customer selection.
The most significant features are presented in the table below: 

![alt text](https://github.com/MichalGrzebyk/credit_card_customers/blob/main/Correlation.png?raw=true)

# Model
I decided to build simple Neural Network:

model = Sequential()
model.add(Dense(128, input_dim=19, activation='relu', kernel_regularizer=regularizers.l2(0.01), kernel_initializer='random_normal'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Training
The dataset has been narrowed down to the strongest features. 
Then the dataset was split into train- and test-dataset (80%/20%).

I decided to use recall, because it is important to limit the prediction of leaving customers as remaining customers.

model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(learning_rate=0.00005), metrics=['Recall'])
model.fit(train_data, train_targets, epochs=2000, batch_size=100, verbose=1)

#Results
The final results on the training set are as follows: 
  - recall: 94.84%
  - accuracy: 80.90%

The final results on the test set are as follows: 
  - recall: 93.88%
  - accuracy: 80.94%

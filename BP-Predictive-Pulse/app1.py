# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("patient_data.csv")


df.rename(columns={'C':'Gender'}, inplace=True)
print(df.info)

# Checking for null values
df.isnull().sum()

# Convert categorical data into numerical
from sklearn.preprocessing import LabelEncoder

columns = ['Gender','Age','History','Patient','TakeMedication','Severity',     'BreathShortness','VisualChanges','NoseBleeding','Whendiagnoused','Systolic','Diastolic','ControlledDiet','Stages']

label_encoder = LabelEncoder()
for col in columns:
    df[col] = label_encoder.fit_transform(df[col])

df['Stages'].replace({'HYPERTENSIVE CRISI':'HYPERTENSIVE CRISIS', 
                      'HYPERTENSION (Stage-2).' : 'HYPERTENSION (Stage-2)'})

df['Stages'].unique()



# DATA ANALYSIS
# Univariate analysis
gender_count = df['Gender'].value_counts()
# plotting the pie chart
plt.pie(gender_count, labels=gender_count.index, autopct='%1.0f%%', startangle=140)
plt.title('Gender Distribution')
plt.axis('equal')
# plt.show()

frequency = df['Stages'].value_counts()
# plotting the bar chart
plt.figure(figsize=(6,6))
frequency.plot(kind='bar')
plt.xlabel('Stages')
plt.ylabel('Frequency')
plt.title('Count of Stages')
# plt.show()

# Bivariate Analysis
sns.countplot(x='TakeMedication', hue='Severity', data=df)
plt.title('Count Plot of TakeMedication by Severity')
# plt.show()

# Multivariate analysis
sns.pairplot(df[['Age','Systolic','Diastolic']])
# plt.show()


# Splitting the data into x and y
X = df.drop('Stages' , axis=1)
Y = df['Stages']

# Splitting into training and testing dataset
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=30)


# Logistic Regression Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

logistic_regression = LogisticRegression()
logistic_regression.fit(x_train, y_train)
y_pred = logistic_regression.predict(x_test)

acc_lr = accuracy_score(y_test, y_pred)
c_lr = classification_report(y_test, y_pred)

print('Accuracy Score : ', acc_lr)
print(c_lr)


# Random Forest Regressor
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier()
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)

acc_rf = accuracy_score(y_test, y_pred)
c_rf = classification_report(y_test, y_pred)

print('Accuracy Score : ', acc_rf)
print(c_rf)


# Decision Tree Model
from sklearn.tree import DecisionTreeClassifier

decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(x_train, y_train)
y_pred = decision_tree_model.predict(x_test)

acc_dt = accuracy_score(y_test, y_pred)
c_dt = classification_report(y_test, y_pred)

print('Accuracy Score : ', acc_dt)
print(c_dt)


# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

NB = GaussianNB()
NB.fit(x_train, y_train)
y_pred = NB.predict(x_test)

acc_nb = accuracy_score(y_test, y_pred)
c_nb = classification_report(y_test, y_pred)

print('Accuracy Score : ', acc_nb)
print(c_nb)


# Multinomial Navies Bayes
from sklearn.naive_bayes import MultinomialNB

mNB = MultinomialNB()
mNB.fit(x_train, y_train)
y_pred = mNB.predict(x_test)

acc_mnb = accuracy_score(y_test, y_pred)
c_mnb = classification_report(y_test, y_pred)

print('Accuracy Score : ', acc_mnb)
print(c_mnb)


prediction = random_forest.predict([[0,3,0,2,0,0,1,6,0,0,0,0,0]])
prediction[0]


# Testing model with multiple evaluation metrics
model = pd.DataFrame({'Model' : ['Linear Regression' , 'Decision Tree Classifier' , 'Random Forest Classifier' , 'Gaussian Naive Bayes' , 'Multinominal Naive Bayes'],
                      'Score' : [acc_lr, acc_dt, acc_rf, acc_nb, acc_mnb]})

print(model)


#  -------------------------------------------------------------------------------
# Save the Best Model
import pickle
import warnings
pickle.dump(random_forest, open("model.pkl", "wb"))


# Build Python Code
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))


# Render HTML Pages
@app.route('/') # route to display the home page
def home():
    return render_template('index.html')


# Retreive the value from UI:
@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == 'POST':
        try:
            Gender = float(request.form["Gender"])
            Age = float(request.form["Age"])
            History = float(request.form["History"])
            Patient = float(request.form["Patient"])
            TakeMedication = float(request.form["TakeMedication"])
            Severity = float(request.form["Severity"])
            BreathShortness = float(request.form["BreathShortness"])
            VisualChanges = float(request.form["VisualChanges"])  # corrected name
            NoseBleeding = float(request.form["NoseBleeding"])
            Whendiagnoused = float(request.form["Whendiagnoused"])
            Systolic = float(request.form["Systolic"])
            Diastolic = float(request.form["Diastolic"])
            ControlledDiet = float(request.form["ControlledDiet"])

            features_values = np.array([[Gender, Age, History, Patient, TakeMedication,
                                         Severity, BreathShortness, VisualChanges,
                                         NoseBleeding, Whendiagnoused,
                                         Systolic, Diastolic, ControlledDiet]])

            df = pd.DataFrame(features_values, columns=[
                'Gender','Age','History','Patient','TakeMedication','Severity',
                'BreathShortness','VisualChanges','NoseBleeding','Whendiagnoused',
                'Systolic','Diastolic','ControlledDiet'
            ])

            prediction = model.predict(df)

            if prediction[0] == 0:
                result = "🟢 NORMAL 🙂"
            elif prediction[0] == 1:
                result = "🟡 HYPERTENSION (Stage-1) ⚠️"
            elif prediction[0] == 2:
                result = "🟠 HYPERTENSION (Stage-2) ⚠️⚠️"
            else:
                result = "🔴 HYPERTENSIVE CRISIS 🚨"

            return render_template("predict.html", prediction_text="Your Blood Pressure stage is: " + result)

        except Exception as e:
            return render_template("predict.html", prediction_text=f"Error: {str(e)}")

    return render_template("predict.html")




@app.route('/details')
def details():
    return render_template('details.html')


if __name__ == "__main__":
    app.run(debug=False, port=5000)



# ----------------------------------------------------------------------------------

# Run Flask in a Desktop App (using PyInstaller)
import webbrowser
from threading import Timer

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == "__main__":
    Timer(1, open_browser).start()  # Wait 1 second before opening
    app.run(debug=False)


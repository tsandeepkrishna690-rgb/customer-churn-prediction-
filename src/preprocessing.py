df=pd.read_csv("customer-churn-project/data/processed/clean_TelcoCustomerChurn.csv")

# Convert TotalCharges safely to numeric, coercing spaces to NaN and filling them with 0
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce').fillna(0)

#feature and traget seperate
X=df.drop("Churn",axis=1)
y=df["Churn"].map({"Yes":1,"No":0})

#train and testing data split in process
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#identify the features dtypes and seperate numbers,objects
numerical_features=X.select_dtypes(include=["int64","float64"]).columns
categorical_features=X.select_dtypes(include="object").columns
print(numerical_features)
print(categorical_features)

#build for features in Columntransformer(numreical and categorical,create preprocessing)
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
scaler=StandardScaler()
encoder=OneHotEncoder(handle_unknown="ignore",drop="first")
preprocessor=ColumnTransformer(transformers=[
    ("num",scaler,numerical_features),
    ("cat",encoder,categorical_features)
])

#train for ml model(classification algorithm)
#create train for pipeline(applies diff preprocessor and ml model)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
log_pipeline=Pipeline(steps=[("proprecessor",preprocessor),("model",LogisticRegression(max_iter=1000))])
log_pipeline.fit(X_train,y_train)
log_prediction=log_pipeline.predict(X_test)
print(log_prediction)
print()

tree_pipeline=Pipeline(steps=[("preprocessor",preprocessor),("model",DecisionTreeClassifier(max_depth=4,random_state=42))])
tree_pipeline.fit(X_train,y_train)
tree_prediction=tree_pipeline.predict(X_test)
print(tree_prediction)
print()

#classification model evaulation and metrics train
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,precision_score,recall_score,f1_score
from sklearn.model_selection import cross_val_score
log_accuracy=accuracy_score(y_test,log_prediction)
tree_accuracy=accuracy_score(y_test,tree_prediction)
log_cm=confusion_matrix(y_test,log_prediction)
tree_cm=confusion_matrix(y_test,tree_prediction)
log_cr=classification_report(y_test,log_prediction)
tree_cr=classification_report(y_test,tree_prediction)
print("log:",log_accuracy)
print("tree:",tree_accuracy)
print("log_cm:",log_cm)
print("tree_cm:",tree_cm)
print("log_cr:",log_cr)
print("tree_cr:",tree_cr)

#train and test performance (gap for both classifications)
log_train_pred=log_pipeline.predict(X_train)
tree_train_pred=tree_pipeline.predict(X_train)
log_test_pred=log_pipeline.predict(X_test)
tree_test_pred=tree_pipeline.predict(X_test)
print(accuracy_score(y_train,log_train_pred))
print(accuracy_score(y_train,tree_train_pred))
print(accuracy_score(y_test,log_test_pred))
print(accuracy_score(y_test,tree_test_pred))

#crossvalidation and hyperparmeter best tunning
from sklearn.model_selection import cross_val_score
log_cv_scores=cross_val_score(log_pipeline,X,y,cv=5,scoring="f1")
print("logcvscore:",log_cv_scores)
print(log_cv_scores.mean())
print(log_cv_scores.std())

from sklearn.model_selection import GridSearchCV
param_grid={
    "model__max_depth":[2,3,4,5,6,8],
    "model__min_samples_split":[2,3,4,5,6,8,10],
    "model__min_samples_leaf":[1,2,5,6,7,8]
}
grid_search=GridSearchCV(
    tree_pipeline,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1)
grid_search.fit(X_train,y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)
print(grid_search.best_estimator_)
#best tunned
best_tree = grid_search.best_estimator_
best_tree_train_pred = best_tree.predict(X_train)
best_tree_test_pred = best_tree.predict(X_test)
best_tree_train_accuracy = accuracy_score(y_train, best_tree_train_pred)
best_tree_test_accuracy = accuracy_score(y_test, best_tree_test_pred)

print("Original Decision Tree")
print("Train Accuracy:", accuracy_score(y_train, tree_train_pred))
print("Test Accuracy :", accuracy_score(y_test, tree_test_pred))

print("\nTuned Decision Tree")
print("Train Accuracy:", best_tree_train_accuracy)
print("Test Accuracy :", best_tree_test_accuracy)
#save to dataset
df.to_csv("customer-churn-project/src/preprocessor_TelcoCustomerChurn.csv",index=False)
#
new_data=pd.DataFrame({
    "gender":["Female"],
    "SeniorCitizen":[1],
    "Partner":["Yes"],
    "Dependents":["No"],
    "tenure":[35],
    "PhoneService":["Yes"],
    "MultipleLines":["No"],
    "InternetService":["Fiber optic"],
    "OnlineSecurity":["Yes"],
    "OnlineBackup":["No"],
    "DeviceProtection":["No"],
    "TechSupport":["No"],
    "StreamingTV":["NO"],
    "StreamingMovies":["No"],
    "Contract":["Month-to-month"],
    "PaperlessBilling":["Yes"],
    "PaymentMethod":["Electronic check"],
    "MonthlyCharges":[100],
    "TotalCharges":[3000]
})
# Fixed: use best model to predict
prediction=best_tree.predict(new_data)
print("Prediction:", prediction)


#Save the tuned model
import joblib

best_tree = grid_search.best_estimator_

joblib.dump(
    best_tree,
    "customer-churn-project/models/customer_churn_model.pkl"
)

print("Model saved successfully!")

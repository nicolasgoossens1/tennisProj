import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_random_forest():
    """
    Evaluate the accuracy of the Random Forest Classifier and display a confusion matrix.
    """
    # Load the trained model
    model = joblib.load('tennis_model.pkl')

    # Load the test data
    X_test = joblib.load('X_test.pkl')
    y_test = joblib.load('y_test.pkl')

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Model Accuracy: {accuracy:.2f}")

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Define class labels
    class_labels = ['Player 2 Wins', 'Player 1 Wins']

    # Plot confusion matrix with labels
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Run the evaluation
if __name__ == "__main__":
    evaluate_random_forest()
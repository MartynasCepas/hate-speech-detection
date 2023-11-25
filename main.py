from app import app
import train_model

if __name__ == '__main__':
    train_model.main() 
    app.run(debug=True)
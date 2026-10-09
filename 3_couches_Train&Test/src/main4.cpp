#include "mylib.h"


int main(int argc, char** argv)
{
    std::cout << "Lancement programme - Reseau de neuronnes a 3 couches profondes" << std::endl;

    int N_train = 200;  // points d'entraînement (inchangé pour rester comparable)
    int N_eval = 2000;  // points de validation et de test
    int n1 = 4;         // nb neurones couche 1
    int n2 = 8;         // nb neurones couche 2
    int n3 = 4;         // nb neurones couche 3

    //std::vector<std::vector<double>> X; // Couche d'entrée 2 neurones
    std::vector<std::vector<double>> X_train, X_test, X_val;
    //std::vector<int> y;
    std::vector<int> y_train, y_test, y_val;

    int choix;

    do
    {
        std::cout << "\nChoisissez un test:" << std::endl;
        std::cout << "1. Spirale (tres difficile)" << std::endl;
        std::cout << "2. XOR (classique non-lineaire)" << std::endl;
        std::cout << "3. Damier (tres complexe)" << std::endl;
        std::cout << "4. Cercles concentriques multiples" << std::endl;
        std::cout << "5. Donnees linéairement separables (facile)" << std::endl;
        std::cout << "6. Croissants de lune (difficulte moyenne)" << std::endl;
        std::cout << "7. Cercles concentriques (dataset original)" << std::endl;
        std::cout << "Votre choix (1-7): ";

        std::cin >> choix;
    } while (choix < 1 || choix > 7);

    Run(choix, X_train, y_train, N_train, 42);    // entrainement
    Run(choix, X_val, y_val, N_eval, 999);     // Validation
    Run(choix, X_test, y_test, N_eval, 1234);    // Test

    std::cout << "Entrainement: " << N_train << " points, Validation/Test: " << N_eval << " points" << std::endl;
    std::cout << "Architecture: " << X_train[0].size() << " -> " << n1 << " -> " << n2 << " -> " << n3 << " -> " << N_OUTPUT << std::endl;
    
    NetworkParams params = Neural_network(X_train, y_train, X_val, y_val, n1, n2, n3, N_OUTPUT, 0.05, 100000);
    
    // Évaluation sur le jeu de test (jamais vu pendant l'entraînement)
    Activations act_test = ForwardPropagation(X_test, params);
    double test_loss = log_loss(act_test.A4, y_test);
    std::vector<int> y_pred_test = Predict(X_test, params);
    double test_acc = Accuracy(y_test, y_pred_test);

    std::cout << "Test -> Loss: " << test_loss << ", Accuracy: " << test_acc << std::endl;
    
    PlotDecisionBoundary(X_test, y_test, params);

    return 0;
}

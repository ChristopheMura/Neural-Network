#include <iostream>
#include <vector>
#include <random>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

struct NetworkParams
{
    std::vector<std::vector<double>> W1;
    std::vector<double> b1;
    std::vector<std::vector<double>> W2;
    std::vector<double> b2;
};

struct Activations
{
    std::vector<std::vector<double>> A1;
    std::vector<double> A2;
    std::vector<std::vector<double>> Z1;
    std::vector<double> Z2;
};

struct Gradients
{
    std::vector<std::vector<double>> dW1;
    std::vector<double> db1;
    std::vector<std::vector<double>> dW2;
    std::vector<double> db2;
};


NetworkParams Initialisation(int n0, int n1, int n2);

Activations ForwardPropagation(std::vector<std::vector<double>>& X, NetworkParams& params);

double log_loss(std::vector<double>& A, std::vector<int> y);

Gradients BackPropagation(std::vector<std::vector<double>>& X, std::vector<int>& y, NetworkParams& params, Activations& activations);

void Update(NetworkParams& params, Gradients& gradients, double learning_rate);

NetworkParams Neural_network(std::vector<std::vector<double>>& X, std::vector<int>& y, int n1, double learning_rate=0.1, int n_iter=100);

std::vector<int> Predict(std::vector<std::vector<double>>& X, NetworkParams& params);

double Accuracy(std::vector<int>& y, std::vector<int>& y_pred);

void PlotDecisionBoundary(std::vector<std::vector<double>>& X, std::vector<int>& y, NetworkParams& params);

int main(int argc, char** argv)
{
    std::cout << "Lancement programme - Reseau de neuronnes a 2 couches" << std::endl;

    int N = 200;
    int n1 = 16;

    std::vector<std::vector<double>> X;
    std::vector<int> y;

    std::random_device rd;
    std::mt19937 gen(42);

    std::uniform_real_distribution<> uniform_dist(0.0, 1.0);
    std::uniform_real_distribution<> angle_dist(0.0, 2.0 * M_PI);
    std::normal_distribution<> noise_dist(0.0, 0.1);
    
    // std::normal_distribution<> dist1_x(2.0, 1.0);
    // std::normal_distribution<> dist1_y(3.0, 1.0);

    // std::normal_distribution<> dist2_x(7.0, 1.0);
    // std::normal_distribution<> dist2_y(6.0, 1.0);

    // Génération des données circulaires
    for (int i = 0; i < N; i++)
    {
        double x_val, y_val;
        double radius = std::sqrt(uniform_dist(gen)); // Distribution uniforme en surface
        double angle = angle_dist(gen);
        
        if (i < N/2)
        {
            // Classe 0: Points au centre (cercle intérieur)
            radius *= 0.5; // Rayon maximum de 0.5
            x_val = radius * std::cos(angle) + noise_dist(gen);
            y_val = radius * std::sin(angle) + noise_dist(gen);
            y.push_back(0);
        }
        else
        {
            // Classe 1: Points en couronne (cercle extérieur)
            radius = 0.6 + radius * 0.4; // Rayon entre 0.6 et 1.0
            x_val = radius * std::cos(angle) + noise_dist(gen);
            y_val = radius * std::sin(angle) + noise_dist(gen);
            y.push_back(1);
        }
        
        X.push_back({x_val, y_val});
    }

    // for (int i=0; i<N/2; i++)
    // {
    //     X.push_back({dist1_x(gen), dist1_y(gen)});
    //     y.push_back(0);
    // }

    // for (int i=0; i<N/2; i++)
    // {
    //     X.push_back({dist2_x(gen), dist2_y(gen)});
    //     y.push_back(1);
    // }

    NetworkParams params = Neural_network(X, y, n1, 0.01, 10000);
    PlotDecisionBoundary(X, y, params);

    return 0;
}

NetworkParams Initialisation(int n0, int n1, int n2)
{
    NetworkParams params;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, 1.0);

    params.W1.resize(n1, std::vector<double>(n0));
    for (int i=0; i<n1; i++)
    {
        for (int j=0; j<n0; j++)
        {
            params.W1[i][j] = dist(gen);
        }
    }
    params.b1.resize(n1, 0.0);

    params.W2.resize(n2, std::vector<double>(n1));
    for (int i=0; i<n2; i++)
    {
        for (int j=0; j<n1; j++)
        {
            params.W2[i][j] = dist(gen);
        }
    }
    params.b2.resize(n2, 0.0);

    return params;
}

Activations ForwardPropagation(std::vector<std::vector<double>>& X, NetworkParams& params)
{
    Activations activations;

    int m = X.size();           // nombre d'échantillons
    int n0 = X[0].size();       // nombre de features
    int n1 = params.W1.size();  // nombre de neuronnes couche 1
    int n2 = params.W2.size();  // nombre de neuronnes couche 2

    // Initialisation des matrices
    activations.Z1.resize(n1, std::vector<double>(m));
    activations.A1.resize(n1, std::vector<double>(m));
    activations.Z2.resize(m);
    activations.A2.resize(m);

    // Calcul Z1 = W1 * X + b1
    for (int i=0; i<n1; i++)
    {
        for (int j=0; j<m; j++)
        {
            activations.Z1[i][j] = params.b1[i];
            for (int k=0; k<n0; k++)
            {
                activations.Z1[i][j] += params.W1[i][k]*X[j][k];
            }
        }
    }

    // Calcul A1 = sigmoid(Z1)
    for (int i=0; i<n1; i++)
    {
        for (int j=0; j<m; j++)
        {
            activations.A1[i][j] = 1.0 / (1.0 + std::exp(-activations.Z1[i][j]));
        }
    }

    // Calcul Z2 = W2 * A1 + b2
    for (int j=0; j<m; j++)
    {
        activations.Z2[j] = params.b2[0];
        for (int k=0; k<n1; k++)
        {
            activations.Z2[j] += params.W2[0][k] * activations.A1[k][j];
        }
    }

    for (int j=0; j<m; j++)
    {
        activations.Z2[j] = 1.0 / (1.0 + std::exp(-activations.Z2[j]));
    }

    return activations;
}

double log_loss(std::vector<double>& A2, std::vector<int> y)
{
    int m = y.size();
    double cost = 0.0;
    double eps = 1e-15;

    for (int i=0; i<m; i++)
    {
        double ai = std::min(std::max(A2[i], eps), 1.0 - eps);
        cost += -y[i] * std::log(ai) - (1 - y[i])*std::log(1 - ai);
    }

    return cost / m;
}

Gradients BackPropagation(std::vector<std::vector<double>>& X, std::vector<int>& y, NetworkParams& params, Activations& activations)
{
    Gradients gradients;

    int m = y.size();
    int n0 = X[0].size();
    int n1 = activations.A1.size();
    int n2 = 1;

    // Initialisation des gradients
    gradients.dW1.resize(n1, std::vector<double>(n0, 0.0));
    gradients.db1.resize(n1, 0.0);
    gradients.dW2.resize(n2, std::vector<double>(n1, 0.0));
    gradients.db2.resize(n2, 0.0);

    // Calcul dZ2 = A2 - y
    std::vector<double> dZ2(m);
    for (int i=0; i<m; i++)
    {
        dZ2[i] = activations.A2[i] - y[i];
    }

    // Calcul dW2 = (1/m) * dZ2 . A1.T
    for (int i=0; i<n2; i++)
    {
        for (int j=0; j<n1; j++)
        {
            gradients.dW2[i][j] = 0.0;
            for (int k=0; k<m; k++)
            {
                gradients.dW2[i][j] += dZ2[k] * activations.A1[j][k];
            }
            gradients.dW2[i][j] /= m;
        }
    }

    // Calcul db2 = (1/m) * sum(dZ2)
    for (int i=0; i<n2; i++)
    {
        gradients.db2[i] = 0.0;
        for (int j=0; j<m; j++)
        {
            gradients.db2[i] += dZ2[j];
        }
        gradients.db2[i] /= m;
    }

    // Calcul dZ1 = W2.T . dZ2 * A1 * (1 - A1)
    std::vector<std::vector<double>> dZ1(n1, std::vector<double>(m));
    for (int i=0; i<n1; i++)
    {
        for (int j=0; j<m; j++)
        {
            // W2.T . dZ2 : somme sur les neuronnes de sortie
            double W2T_dZ2 = 0.0;
            for (int k=0; k<n2; k++)
            {
                W2T_dZ2 += params.W2[k][i] * dZ2[j];
            }
            // Multiplication par la dérivée de sigmoid : A1 * (1 - A1)
            dZ1[i][j] = W2T_dZ2 * activations.A1[i][j] * (1 - activations.A1[i][j]);
        }
    }

    // Calcul dW1 = (1/m) * dZ1 . X.T
    for (int i=0; i<n1; i++)
    {
        for (int j=0; j<n0; j++)
        {
            gradients.dW1[i][j] = 0.0;
            for (int k=0; k<m; k++)
            {
                gradients.dW1[i][j] += dZ1[j][k] * X[k][j];
            }
            gradients.dW1[i][j] /= m;
        }
    }

    // Calcul db1 = (1/m) * sum(dZ1, axis=1)
    for (int i=0; i<n1; i++)
    {
        gradients.db1[i] = 0.0;
        for (int j=0; j<m; j++)
        {
            gradients.db1[i] += dZ1[i][j];
        }
        gradients.db1[i] /= m;
    }

    return gradients;
}

void Update(NetworkParams& params, Gradients& gradients, double learning_rate)
{
    int n0 = params.W1[0].size();
    int n1 = params.W1.size();
    int n2 = params.W2.size();

    // Mise à jour W1 = W1 - learning_rate * dW1
    for (int i=0; i<n1; i++)
    {
        for (int j=0; j<n0; j++)
        {
            params.W1[i][j] = params.W1[i][j] - learning_rate * gradients.dW1[i][j];
        }
    }

    // Mise à jour b1 = b1 - learning_rate * db1
    for (int i=0; i<n1; i++)
    {
        params.b1[i] = params.b1[i] - learning_rate * gradients.db1[i];
    }

    // Mise à jour W2 = W2 - learning_rate * dW2
    for (int i=0; i<n2; i++)
    {
        for (int j=0; j<n1; j++)
        {
            params.W2[i][j] = params.W2[i][j] - learning_rate * gradients.dW2[i][j];
        }
    }

    // Mise à jour b2 = b2 - learning_rate * db2
    for (int i=0; i<n2; i++)
    {
        params.b2[i] = params.b2[i] - learning_rate * gradients.db2[i];
    }
}

NetworkParams Neural_network(std::vector<std::vector<double>>& X, std::vector<int>& y, int n1, double learning_rate, int n_iter)
{
    // Initialisation des paramètres
    int n0 = X[0].size();   // nombre de features d'entrée
    int n2 = 1;             // nombre de sorties

    NetworkParams params = Initialisation(n0, n1, n2);

    // Historiques pour les courbes d'apprentissage
    std::vector<double> train_loss;
    std::vector<double> train_acc;

    std::cout << "Debut de l'entrainement du reseau de neuronnes..." << std::endl;

    // Decente de Gradient
    for (int i=0; i<n_iter; i++)
    {
        Activations activations = ForwardPropagation(X, params);

        // Calcul de la perte
        double loss = log_loss(activations.A2, y);
        train_loss.push_back(loss);

        // Calcul de la précision
        std::vector<int> y_pred = Predict(X, params);
        double acc = Accuracy(y, y_pred);
        train_acc.push_back(acc);

        // Affichage des progrès
        if (i%100 == 0)
        {
            std::cout << "Iteration " << i << ", Loss: " << loss << ", Accuracy: " << acc << std::endl;
        }

        // Backward propagation
        Gradients gradients = BackPropagation(X, y, params, activations);

        // Mise à jour des paramètres
        Update(params, gradients, learning_rate);
    }

    // Affichage final
    std::cout << "Entrainement termine !" << std::endl;
    std::cout << "Loss finale: "<<  train_loss.back() << std::endl;
    std::cout << "Precision finale: " << train_acc.back() << std::endl;

    // // Affichage des courbes d'apprentissage
    // plt::figure_size(1200, 400);

    // // Subplot 1: Loss
    // plt::subplot(1, 2, 1);
    // plt::plot(train_loss, {{"label", "train loss"}});
    // plt::xlabel("Itérations");
    // plt::ylabel("Loss");
    // plt::title("Évolution de la perte");
    // plt::legend();
    // plt::grid(true);

    // // Subplot 2: Accuracy
    // plt::subplot(1, 2, 2);
    // plt::plot(train_acc, {{"label", "train acc"}});
    // plt::xlabel("Itérations");
    // plt::ylabel("Accuracy");
    // plt::title("Évolution de la précision");
    // plt::legend();
    // plt::grid(true);

    // plt::show();

    return params;
}

std::vector<int> Predict(std::vector<std::vector<double>>& X, NetworkParams& params)
{
    // Appel de la forward propagation
    Activations activations = ForwardPropagation(X, params);

    // Récupération de A2
    std::vector<double>& A2 = activations.A2;

    // Prédiction : A2 >= 0.5
    std::vector<int> prediction(A2.size());
    for (size_t i=0; i<A2.size(); i++)
    {
        prediction[i] = (A2[i] >= 0.5) ? 1 : 0;
    }

    return prediction;
}

double Accuracy(std::vector<int>& y, std::vector<int>& y_pred)
{
    int correct = 0;

    for (size_t i=0; i<y.size(); i++)
    {
        if (y[i] == y_pred[i])
        {
            correct += 1;
        }
    }

    return (double)correct / y.size();
}

// void PlotDecisionBoundary(std::vector<std::vector<double>>& X, std::vector<int>& y, NetworkParams& params)
// {
//     // Séparation des points par classe
//     std::vector<double> x0, y0, x1, y1;

//     for (size_t i = 0; i < X.size(); i++)
//     {
//         if (y[i] == 0)
//         {
//             x0.push_back(X[i][0]);
//             y0.push_back(X[i][1]);
//         }
//         else
//         {
//             x1.push_back(X[i][0]);
//             y1.push_back(X[i][1]);
//         }
//     }

//     // Affichage des points
//     plt::figure();
//     plt::scatter(x0, y0, 20.0, {{"color", "green"}, {"label", "Classe 0"}});
//     plt::scatter(x1, y1, 20.0, {{"color", "blue"}, {"label", "Classe 1"}});

//     // Création d'une grille pour la frontière de décision
//     double x_min = 0.0, x_max = 10.0;
//     double y_min = 0.0, y_max = 10.0;
//     int n_points = 50;

//     std::vector<std::vector<double>> grid_points;
//     std::vector<double> grid_x, grid_y;

//     for (int i = 0; i < n_points; i++)
//     {
//         for (int j = 0; j < n_points; j++)
//         {
//             double x_val = x_min + (x_max - x_min) * i / (n_points - 1);
//             double y_val = y_min + (y_max - y_min) * j / (n_points - 1);
//             grid_points.push_back({x_val, y_val});
//             grid_x.push_back(x_val);
//             grid_y.push_back(y_val);
//         }
//     }

//     // Prédictions sur la grille
//     std::vector<int> grid_predictions = Predict(grid_points, params);

//     // Affichage des contours (approximation avec des points colorés)
//     std::vector<double> boundary_x, boundary_y;
//     for (size_t i = 0; i < grid_predictions.size(); i++)
//     {
//         Activations activations = ForwardPropagation(grid_points, params);
//         // Points proches de la frontière
//         if (std::abs(activations.A2[i] - 0.5) < 0.05)
//         { 
//             boundary_x.push_back(grid_x[i]);
//             boundary_y.push_back(grid_y[i]);
//         }
//     }

//     if (!boundary_x.empty())
//     {
//         plt::scatter(boundary_x, boundary_y, 1.0, {{"color", "red"}, {"alpha", "0.3"}});
//     }

//     plt::xlabel("Feature 1");
//     plt::ylabel("Feature 2");
//     plt::title("Réseau de neurones - Frontière de décision");
//     plt::legend();
//     plt::grid(true);
//     plt::show();
// }


void PlotDecisionBoundary(std::vector<std::vector<double>>& X, std::vector<int>& y, NetworkParams& params)
{
    // Séparation des points par classe
    std::vector<double> x0, y0, x1, y1;

    for (size_t i = 0; i < X.size(); i++)
    {
        if (y[i] == 0)
        {
            x0.push_back(X[i][0]);
            y0.push_back(X[i][1]);
        }
        else
        {
            x1.push_back(X[i][0]);
            y1.push_back(X[i][1]);
        }
    }

    // Affichage des points de données
    plt::scatter(x0, y0, 40.0, {{"color", "yellow"}});
    plt::scatter(x1, y1, 40.0, {{"color", "teal"}});

    // Création d'une grille pour la frontière de décision
    double x_min = -1.5, x_max = 1.5;
    double y_min = -1.5, y_max = 1.5;
    int n_points = 100;

    // Calcul des probabilités sur la grille et détection de la frontière
    std::vector<double> boundary_x, boundary_y;
    
    for (int i = 0; i < n_points-1; i++)
    {
        for (int j = 0; j < n_points-1; j++)
        {
            double x1_val = x_min + (x_max - x_min) * i / (n_points - 1);
            double y1_val = y_min + (y_max - y_min) * j / (n_points - 1);
            double x2_val = x_min + (x_max - x_min) * (i+1) / (n_points - 1);
            double y2_val = y_min + (y_max - y_min) * (j+1) / (n_points - 1);
            
            // Test des 4 points du carré
            std::vector<std::vector<double>> test_points = {
                {x1_val, y1_val}, {x2_val, y1_val}, 
                {x1_val, y2_val}, {x2_val, y2_val}
            };
            
            std::vector<int> predictions;
            for (auto& point : test_points)
            {
                std::vector<std::vector<double>> single_point = {point};
                std::vector<int> pred = Predict(single_point, params);
                predictions.push_back(pred[0]);
            }
            
            // Si les prédictions changent dans ce carré, on est sur la frontière
            bool has_boundary = false;
            for (size_t k = 1; k < predictions.size(); k++)
            {
                if (predictions[k] != predictions[0])
                {
                    has_boundary = true;
                    break;
                }
            }
            
            if (has_boundary)
            {
                // Ajouter le centre du carré comme point de frontière
                boundary_x.push_back((x1_val + x2_val) / 2.0);
                boundary_y.push_back((y1_val + y2_val) / 2.0);
            }
        }
    }

    // Affichage de la frontière de décision
    if (!boundary_x.empty())
    {
        plt::scatter(boundary_x, boundary_y, 10.0, {{"color", "red"}});
    }

    plt::show();
}
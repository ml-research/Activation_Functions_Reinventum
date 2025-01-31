import json

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
import torch.nn as nn



class RationalsModel(torch.nn.Module):
    def __init__(self, n=5, m=5, function="gauss", use_coefficients=False):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the attributes of an instance of a class.

        :param self: Represent the instance of the class
        :param n: Set the number of coefficients of numerator
        :param m: Set the number of coefficients of denominator
        :param function: Specify the activation function
        :param use_coefficients: Determine if the coefficients should be loaded
        from a file or not
        """
        super().__init__()
        self.n = n
        self.m = m
        self.degree = n + m
        self.func_name = function
        self.function = self.load_func(func=function)
        if use_coefficients:
            self.load_coefficients()
        else:
            if function == "uniform":
                self.coeff_numerator = nn.Parameter(torch.rand(self.m, requires_grad=True))
                self.coeff_denominator = nn.Parameter(torch.rand(self.n - 1, requires_grad=True))
            elif function == "gauss":
                self.coeff_numerator = nn.Parameter(torch.normal(mean=0.0, std=1.0, size=(self.m,), requires_grad=True))
                self.coeff_denominator = nn.Parameter(torch.normal(mean=0.0, std=1.0, size=(self.n-1,), requires_grad=True))
            else:
                raise ValueError(f"Unsupported init function, expected 'uniform' or 'gauss' got {function}")

        self.function_list = ["relu", "sigmoid", "tanh", "leaky_relu", "swish", "gelu"]
        self.float()

    def forward(self, x):
        """
        The forward function computes the polynomial given by:

        :param self: Represent the instance of the class
        :param x: Compute the polynomial
        :return: A tensor with same shape as x
        """
        x_powers = torch.pow(x[..., None], torch.arange(max(self.m, self.n+1), device=x.device))
        
        # torch.matmul will decide what operation to use:
        # - if x is 1D, dot product will be compute
        # - if x is 2D, torch.mm will be computed
        # - if x is ND (N>2), batched matrix multiplication (torch.bmm) over all but last two dimensions
        #   will be computed
        # If necessary torch.matmul will append a dimension of size one to second matrix (here numerator/denominator).
        numerator = torch.matmul(x_powers[..., :self.m], self.coeff_numerator)
        denominator = torch.matmul(x_powers[..., 1:self.n], self.coeff_denominator)
        # compute overall polynomial
        polynomial = numerator.div(torch.abs(denominator) + 1)

        return polynomial

    def _train(
        self, name, x_train, y_train, learning_rate=0.01, num_epochs=4000, render=False
    ):
        """
        The train function takes in a name, x_train, y_train, learning rate (default 0.01),
        number of epochs (default 10000), and render flag (default False). The function then
        creates an optimizer using the Adam optimizer with the coefficients as parameters and
        a loss function using MSE. It also initializes some lists to store values for plotting
        later on. The update plot function is defined here as well which will be used by
        FuncAnimation to animate our plots later on. Then we loop through each epoch and
        calculate our predicted y value based off of our current coefficients from forward().

        :param self: Refer to the class instance itself
        :param name: Refers to the function neme
        :param x_train: the considered space for the training
        :param y_train: the function to train
        :param learning_rate: control the speed of learning
        :param num_epochs: Set the number of iterations that the model will train for
        :param render: Render the animation of the training process
        :return: The loss of the last epoch
        """

        optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=0.01
        )
        loss_fn = torch.nn.MSELoss()
        end_loss = 0.0

        fig, ax = plt.subplots()
        y_pred_values = []  # Store y_pred values for plotting
        epoch_values = []  # Store epoch values for plotting
        predictions = []
        true_functions = []
        upper_bound = 10
        lower_bound = 0.0009

        def update_plot(epoch):
            ax.clear()
            ax.plot(x_train.cpu().numpy(), predictions[epoch], label=f"predict {name}")
            ax.plot(x_train.cpu().numpy(), true_functions[epoch], label=f"true {name}")
            ax.legend()
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"Epoch {epoch_values[epoch]}")

        for epoch in range(num_epochs):
            y_pred = self.forward(x_train)
            loss = loss_fn(y_pred, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")
                predictions.append(y_pred.cpu().detach().numpy())
                true_functions.append(y_train.cpu().numpy())
                epoch_values.append(epoch)

            end_loss = loss.item()
            if end_loss > upper_bound or end_loss < lower_bound:
                print(f"Epoch {epoch}, Loss: {loss.item()}")
                break

        self.store_coefficients()

        ani = animation.FuncAnimation(
            fig, update_plot, frames=len(epoch_values), interval=1
        )
        if render:
            ani.save(
                f"rational_trained_models/{name}_function.gif",
                writer="ffmpeg",
                fps=30,
            )
            plt.close()

        return end_loss

    def load_func(self, func="relu", range=2):
        """
        The load_func function takes in a string and returns the corresponding
        activation function.
            Args:
                func (str): The name of the activation function to be returned.

        :param range: range to define the range of the space
        :param self: Represent the instance of the class
        :param func: Specify the activation function to be used
        :param space: Define the range of values that will be passed to the
        activation function
        :return: The activation function that is passed to it
        """
        space = torch.linspace(-range, range, 100)
        if func == "relu":
            true_func = torch.nn.functional.relu(space)
        elif func == "sigmoid":
            true_func = torch.nn.functional.sigmoid(space)
        elif func == "tanh":
            true_func = torch.nn.functional.tanh(space)
        elif func == "leaky_relu":
            true_func = torch.nn.functional.leaky_relu(space)
        elif func == "swish":
            true_func = torch.nn.functional.silu(space)
        elif func == "gelu":
            true_func = torch.nn.functional.gelu(space)
        else:
            true_func = torch.nn.functional.relu(space)
        return true_func

    def _show(self, x, y, func=None):
        """
        The _show function is a helper function that plots the true and
        predicted functions. It takes in three arguments: x_true, y_true,
        and func. The first two are the data points that we want to plot
        on our graph (the true values). The third argument is optional; it's
        a string that will be used as a label for the legend of our graph.

        :param self: Access the attributes and methods of the class
        :param x: Space of function
        :param y: the function
        :param func: Label the plot
        :return: A plot of the true function and the predicted function
        """
        y_pred = self.forward(x)
        plt.plot(x, y, label=f"True {func}")
        plt.plot(x, y_pred.cpu().detach().numpy(), label=f"Predicted {func}")
        plt.legend()
        plt.show()

    def show_all(self, range=2):
        """
        The show_all function takes in a list of functions and plots the
        true function, the trained model's approximation, and the error
        between them. It also prints out the mean squared error for each
        function.

        :param range: range to define the range of the space
        :param self: Access the attributes of the class
        :return: A plot of the functions and their approximations
        """
        space = torch.linspace(-range, range, 100)

        for func in self.function_list:
            name = f"rational_trained_models/{func}_model.pt"
            model = torch.load(name)
            y = model.load_func(func, range=range)
            model._show(space, y, func)

    def show(self, range=2):
        """
        The show function takes the trained model and plots it against
        the true function. It also prints out a table of values for both
        functions at various points.

        :param range: range to define the range of the space
        :param self: Access the attributes and methods of the class
        :return: A plot of the function, and the predicted values
        """
        space = torch.linspace(-range, range, 100)
        func = self.func_name
        name = f"rational_trained_models/{func}_model.pt"
        model = torch.load(name)
        y = model.load_func(func)
        model._show(space, y, func)

    def store_coefficients(self):
        """
        The store_coefficients function takes the coefficients of a rational function
        and stores them in a JSON file. The name of the JSON file is based on the name
        of the function that was used to generate it. This allows us to store multiple
        trained models for different functions, and then load them later.

        :param self: Refer to the object itself
        :return: A dictionary with the function name and coefficients of numerator and
        denominator
        """
        coefficients_dict = {
            "function": self.func_name,
            "coeff_numerator": self.coeff_numerator.tolist(),
            "coeff_denominator": self.coeff_denominator.tolist(),
        }
        with open(f"rational_trained_models/coeff_{self.func_name}.json", "w") as file:
            json.dump(coefficients_dict, file, indent=1, separators=(", ", " : "))

    def load_coefficients(self):
        """
        The load_coefficients function loads the coefficients of a trained model
        from a json file. The function takes no arguments and returns nothing.
        The function will raise an error if the  json file does not exist or if
        it contains data for another function.

        :param self: Refer to the object itself
        :return: The coefficients of numerator and denominator for the function
        name given
        """
        try:
            with open(
                f"rational_trained_models/coeff_{self.func_name}.json", "r"
            ) as file:
                data = json.load(file)
                if data["function"] == self.func_name:
                    self.coeff_numerator = nn.Parameter(
                        torch.tensor(data["coeff_numerator"], requires_grad=True)
                    )
                    self.coeff_denominator = nn.Parameter(
                        torch.tensor(data["coeff_denominator"], requires_grad=True)
                    )
                else:
                    raise ValueError("No function found!")
        except FileNotFoundError:
            print("No file found!")
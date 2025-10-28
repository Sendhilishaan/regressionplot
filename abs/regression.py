class Regression:
    """
    abstract class for a regression model 
    """
    def __init__(self, file: str) -> None:
        self._file = file

    def read_csv(self):
        raise NotImplementedError

    def gradient_descent_step(self) -> int:
        """one iteration of gradient descent"""
        raise NotImplementedError

    def cost_function(self, x: int, y: int) -> int:
        """MSE cost function"""
        raise NotImplementedError

    def display_graph(self):
        raise NotImplementedError
import torch


class ADAMGradientStabilization(torch.nn.Module):


    def __init__(self, beta_1: float, beta_2: float, eps: float, reset_step: int):
        '''
        Applies gradient stabilization from ADAM.
        '''
        super().__init__()
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.m = None
        self.v = None
        self.step = 1
        self.reset_step = reset_step


    def __call__(self, classifier_gradient):

        if self.m is None or self.step == self.reset_step + 1:
            self.m = torch.zeros_like(classifier_gradient)
            self.v = torch.zeros_like(classifier_gradient)
            self.step = 1
            # log.info("Resetting stabilization")

        m = self.beta_1 * self.m + (1 - self.beta_1) * classifier_gradient
        self.m = m
        v = self.beta_2 * self.v + (1 - self.beta_2) * torch.square(classifier_gradient)
        self.v = v
        m_hat = m / (1 - (self.beta_1**self.step))
        v_hat = v / (1 - (self.beta_2**self.step))
        self.step += 1

        return m_hat / (torch.sqrt(v_hat) + self.eps)
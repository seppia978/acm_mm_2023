from collections.abc import Iterable
import torch

Tensor = torch.Tensor

class LossManager:

    def __init__(
        self,
        loss_type:str = None,
        logits:bool = False,
        lambdas:Iterable = [],
        classification_loss_fn:type = torch.nn.CrossEntropyLoss,
        reduction=False
    ) -> None:

        self.ltype, self.logits = loss_type, logits
        self.lambdas = lambdas
        self.cls_loss_fn = classification_loss_fn(reduction='none') if not reduction else \
                            classification_loss_fn()

    def classification_loss_selector(self, *args, device = 'cuda') -> Tensor :
        
        # * args should be (scores, imgs, labels, standard_model)
        assert len(args) == 4, f'Loss requries exactly 4 arguments (scores, imgs, labels, standard_model). Found {len(args)}.'

        if self.logits:
            general = args[-1]
            if general is not None:
                general.arch.cuda().eval()
                with torch.no_grad():
                    general_score = general(args[1])
            return self.cls_loss_fn(args[0].to(device), general_score.to(device))
        else:
            return self.cls_loss_fn(args[0].to(device),args[2].to(device))

    def loss_selector(self, **kwargs) -> Tensor :

        assert 'kept_loss' in kwargs.keys(), f'kept_loss missing.'
        assert 'unlearnt_loss' in kwargs.keys(), f'unlearnt_loss missing.'

        kept_loss = kwargs['kept_loss']
        unlearnt_loss = kwargs['unlearnt_loss']

        if 'alpha_norm' in kwargs.keys():
            alpha_norm = kwargs['alpha_norm']

        if self.ltype == 'sum':
                    loss_train = loss_cls.mean()
                    loss_train += alpha_norm
        elif '3way' in self.ltype:
            keep = kept_loss.clone()
            unlearn = unlearnt_loss.clone()
            if 'multiplication' in self.ltype:
                loss_cls = (self.lambdas[0] * keep.mean() / (self.lambdas[1] * torch.abs(unlearn.mean())))
                loss_train = loss_cls + alpha_norm
            elif 'sum' in self.ltype:
                loss_cls = torch.pow(self.lambdas[0] * keep.mean(),2) + torch.pow(self.lambdas[1]/unlearn.mean(),1)
                loss_train = loss_cls + alpha_norm

            elif 'third' in self.ltype:
                loss_cls = self.lambdas[0] * keep.mean() * (1 - self.lambdas[1] / torch.abs(unlearn.mean())) # l+ * (1 - lambda1/l-)
                loss_train = loss_cls + alpha_norm
        else:
            loss_train = loss_cls.mean()

        return loss_train
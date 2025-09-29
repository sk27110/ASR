from torch import Tensor
from torch.nn import CTCLoss

class CTCLossWrapper(CTCLoss):
    def forward(
        self,
        log_probs: Tensor,
        targets: Tensor,
        input_lengths: Tensor,
        target_lengths: Tensor,
        **batch,                    
    ) -> Tensor:
        log_probs_t = log_probs.transpose(0, 1)

        loss = super().forward(
            log_probs_t, targets, input_lengths, target_lengths
        )
        return {"loss": loss}

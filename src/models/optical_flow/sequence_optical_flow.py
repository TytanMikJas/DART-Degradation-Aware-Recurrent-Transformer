import torch

from src.models.optical_flow.optical_flow_handler import OpticalFlowHandlerAbs


class SequenceOpticalFlow(OpticalFlowHandlerAbs):
    def forward(
        self,
        frames: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute bidirectional optical flows for an arbitrary-length video sequence.

        Used by VideoBackbone for processing variable-length sequences.
        Uses the denormalizer for consistent preprocessing.

        Args:
            frames: Video frames (N, T, C, H, W) in normalized range

        Returns:
            Tuple of (forward_flow, backward_flow):
                - forward_flow: Flow from frame i+1 to frame i (N, T-1, 2, H, W)
                - backward_flow: Flow from frame i to frame i+1 (N, T-1, 2, H, W)
        """
        n, t, c, h, w = frames.size()

        if self.flow_net is None:
            # Return zero flows
            forward_flow = frames.new_zeros(n, t - 1, 2, h, w)
            backward_flow = frames.new_zeros(n, t - 1, 2, h, w)
            return forward_flow, backward_flow

        # Reshape all adjacent frame pairs for batched processing
        forward_frames = frames[:, 1:, :, :, :].reshape(-1, c, h, w)  # frames i+1
        backward_frames = frames[:, :-1, :, :, :].reshape(-1, c, h, w)  # frames i

        # Denormalize and convert to [0, 255] range for RAFT (consistent with forward())
        forward_frames_255 = (
            torch.clamp(self.denormalizer(forward_frames), 0.0, 1.0) * 255.0
        )
        backward_frames_255 = (
            torch.clamp(self.denormalizer(backward_frames), 0.0, 1.0) * 255.0
        )

        with torch.cuda.amp.autocast(enabled=False):
            forward_frames_255 = forward_frames_255.float()
            backward_frames_255 = backward_frames_255.float()

            # Compute flow with proper gradient handling based on freeze flag
            # BUG FIX: Conditional no_grad allows gradients when unfrozen
            with torch.set_grad_enabled(not self.freeze):
                forward_flow = self.flow_net(forward_frames_255, backward_frames_255)
                backward_flow = self.flow_net(backward_frames_255, forward_frames_255)

        # Reshape back to sequence format
        forward_flow = forward_flow.view(n, t - 1, 2, h, w)
        backward_flow = backward_flow.view(n, t - 1, 2, h, w)

        return forward_flow, backward_flow

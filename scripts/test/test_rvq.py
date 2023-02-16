import torch
from vector_quantize_pytorch import ResidualVQ

print('running rvq')
rq = ResidualVQ(dim=512, num_quantizers=12, codebook_size=1024, commitment_weight=0, decay=0.95, kmeans_init=True, threshold_ema_dead_code=0)

# rq.load_state_dict(torch.load('./results/semantic/semantic.conditioner_rvq.6000.pt', map_location='cpu'))

for i in range(10):
    q, i, loss = rq(torch.randn(1, 512))
    print(i[0:2])

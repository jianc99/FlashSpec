# import torch

# BATCH_SIZE = 4
# gamma = 4
# tokens_buffer = torch.arange(BATCH_SIZE * (gamma + 1)).view(BATCH_SIZE, gamma + 1).to(torch.long).cuda()
# output = torch.zeros((BATCH_SIZE, 10), device='cuda').long()
# accept_nums = torch.tensor([3, 2, 5, 4], device='cuda').long()  # shape (BATCH_SIZE,)
# offset = torch.tensor([1, 2, 3, 4], device='cuda').long()  # shape (BATCH_SIZE,)

# # Create a mask for the positions to fill in the output tensor
# positions = torch.arange(10, device='cuda').view(1, -1).repeat(BATCH_SIZE, 1)
# mask = (positions < (offset + accept_nums).view(-1, 1)) & (positions >= offset.view(-1, 1))

# positions_buffer = torch.arange(gamma+1, device='cuda').view(1, -1).repeat(BATCH_SIZE, 1)
# mask_buffer = positions_buffer<accept_nums.view(-1,1)


# output[mask] = tokens_buffer[mask_buffer]
# print(tokens_buffer)
# print(output)



# from transformers.models.llama.modeling_llama import(
#     LlamaRMSNorm,
#     LlamaConfig,
#     PreTrainedModel,
#     repeat_kv,
#     ACT2FN
# )

# print(LlamaConfig.from_pretrained("JackFram/llama-68m"))


import torch

# Sample input values
BATCH_SIZE = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize sample inputs
tokens_buffer = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], device=DEVICE).long()
accept_nums = torch.tensor([[2], [1], [3], [4]], device=DEVICE).long()
bonus_tokens = torch.tensor([[17], [18], [19], [20]], device=DEVICE).long()

# args object with gamma value
class Args:
    gamma = 3

args = Args()

# Initialize the tensor 'next' with zeros
next = torch.zeros((BATCH_SIZE, 2), device=DEVICE).long()

# Create a mask for rows where accept_nums == args.gamma + 1
mask = (accept_nums == (args.gamma + 1)).squeeze()

# Apply the conditions to fill the 'next' tensor
next[:, 0] = torch.where(mask, tokens_buffer[:, -1], bonus_tokens[:, 0])
next[:, 1] = torch.where(mask, bonus_tokens[:, 0], torch.zeros_like(bonus_tokens[:, 0]))

# If needed, ensure the values in next[:, 1] are set to 0 when the condition is not met
next[~mask, 1] = 0

# Print the result
print("tokens_buffer:\n", tokens_buffer)
print("accept_nums:\n", accept_nums)
print("bonus_tokens:\n", bonus_tokens)
print("next:\n", next)

non_zero_mask = next != 0
result = non_zero_mask.sum(dim=1, keepdim=True) - 1
print(next.gather(1, result))

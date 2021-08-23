import torch

model = torch.load("best_model.pt")
user_data = torch.tensor([23,34.4,0,0,0,1,0,0,0,1])

with torch.no_grad():
    price = model.forward(user_data)
    print(f"£ {round(price.item(), 2)}")

# def questionnaire():
#     user_data = []
#     input
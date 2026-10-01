import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

ea_ccpl = event_accumulator.EventAccumulator('./logs/')
ea_ccpl.Reload()  # Load the events from the file
ea_NDNSM = event_accumulator.EventAccumulator('./logs/')
ea_NDNSM.Reload()  # Load the events from the file

# Get scalar values (e.g., loss, accuracy)
loss_ccp = ea_ccpl.Scalars('loss_ccp')
loss_ccp_NDNSM = ea_NDNSM.Scalars('loss_ccp')
# for scalar in scalars:
#     print(scalar.step, scalar.value)
steps = [scalar.step for scalar in loss_ccp]
loss_ccp_values = [scalar.value/5 for scalar in loss_ccp]
loss_ccp_NDNSM_values = [scalar.value for scalar in loss_ccp_NDNSM]

plt.figure(dpi=300)
plt.plot(steps, loss_ccp_values, label='CCPL', color='#1E90FF')
plt.plot(steps, loss_ccp_NDNSM_values, label='Ours', color='#FF7F50')
plt.xlabel('Steps', fontsize=14)
plt.ylabel('Lccp', fontsize=14)
plt.title('Lccp of each method', fontsize=14)
plt.legend()
plt.show()

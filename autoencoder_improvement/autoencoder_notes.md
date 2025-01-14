## Inputs
We have a mix of binary and semi-continuous data.

### The Analog Inputs
The analog sticks take values between $[-1, -0.2875]\cup\{0\}\cup[0.2875, 1]$ in increments of .0125. There is a deadzone around 0. The values are not uniformly distributed, there is more values at the edges of the circle. The distribution of the values is shown below (we remove the (0,0) value so that the other values show up on the plots).
![Analog Sticks](stick_hist.png)
We transform the analog inputs in the TrainingDataset class to evenly spaced in the interval $[-1, 1]$.
```python
# Shift inputs closer to 0, respecting the increments of .0125
analog_transformed[analog_transformed > 0] -= 0.2875 + 0.0125
analog_transformed[analog_transformed < 0] += 0.2875 - 0.0125
# Scale inputs to be between -.5 and .5
analog_transformed *= .5 / .725
# Add .5 to so final inputs are between 0 and 1
analog_transformed += .5
```
When we asses the model's performance, we care that the model gets within the bins of the analog inputs or how many bins we are away from the target.
```python
integer_stick_targets = np.round(target[:,0:4] / 0.008620689655172415 ).astype(np.int32)
integer_stick_pred = np.round(pred[:,0:4] / 0.008620689655172415).astype(np.int32)
```
As an example, here we have a table that shows the percent of frames the model was within n bins of the target.
| How Close | JSTICK_X   | JSTICK_Y   | CSTICK_X   | CSTICK_Y   |
|-----------|------------|------------|------------|------------|
| 0         | 55.299356  | 55.299356  | 55.299356  | 55.299356  |
| 1         | 81.331066  | 81.331066  | 81.331066  | 81.331066  |
| 2         | 91.304177  | 91.304177  | 91.304177  | 91.304177  |
| 3         | 94.986024  | 94.986024  | 94.986024  | 94.986024  |
| 4         | 96.817156  | 96.817156  | 96.817156  | 96.817156  |
| 5         | 97.913711  | 97.913711  | 97.913711  | 97.913711  |
| 6         | 98.605853  | 98.605853  | 98.605853  | 98.605853  |
| 7         | 99.048774  | 99.048774  | 99.048774  | 99.048774  |
| 8         | 99.337399  | 99.337399  | 99.337399  | 99.337399  |
| 9         | 99.528090  | 99.528090  | 99.528090  | 99.528090  |

We did try adding additional binary features corresponding to when the analog inputs were 0. This did not improve the model's performance **SHOULD CHECK THIS**
### The Digital Inputs
The digital inputs are binary. We care that the model gets the correct value for these inputs.

 The buttons values are 0 (inactive) far more often than they are 1 (active).
| Button | Active Frames |
|-------|-------------------------------|
| TRIGGER_LOGICAL | 17.22% **(check)** |
| Z | 0.84% |
| A | 6.1% |
| B | 4.59% |
| X_or_Y | 9.8% |

The model's accuracy when the button was not pressed was higher than when the button was pressed.

| Button          | Accuracy   | Acc of 0   | Acc of 1   |
|-----------------|------------|------------|------------|
| TRIGGER_LOGICAL | 99.503160  | 99.969916  | 97.178167  |
| Z               | 99.892392  | 99.997126  | 88.544916  |
| A               | 99.130744  | 99.977962  | 86.338374  |
| B               | 99.599759  | 99.989392  | 91.247724  |
| X_or_Y          | 97.985788  | 99.927363  | 80.251771  |

We tried a weighted binary cross entropy loss to see if this would improve the model's performance **(SHOULD CHECK THIS AGAIN)**.


### The loss function
We tried a variety of different loss functions. The simplest used mean squared error for the analog inputs and binary cross entropy for the binary inputs. 
```python
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.BCE = nn.BCEWithLogitsLoss(reduction='sum')  # Consider using weighted BCE if needed
        self.MSE = nn.MSELoss(reduction='sum')
        
    def forward(self, pred, target):
        # Calculating losses
        mse_loss = self.MSE(torch.sigmoid(pred[:,0:4,0:60]), target[:,0:4,0:60]) 
        bce_loss = self.BCE(pred[:,4:,0:60], target[:,4:,0:60])
        
        # Total loss, we divide the bce loss by 100 to make it comparable to the mse loss
        return mse_loss  + bce_loss / 100
```
Depending on how much we scale the binary cross entropy loss, the model will prioritize the analog or binary inputs. 

**should try a weighted binary cross entropy loss**

## Training
- For some reason, the model did not train when we used autocast. (I think that ``anomaly_detection_minute/convolutional_autoencoder_train_test.ipynb`` is the best version of the train_model function. The one in autoencoder_improvement uses autocast)
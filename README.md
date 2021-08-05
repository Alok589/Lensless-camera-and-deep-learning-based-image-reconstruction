# A hybrid deep convolution neural network (RSE-UNet) 
# Residual block from ResNet and Squeeze and Excitation block from SENet has been used.
# Residual block allows training deeper networks and prevent from vanishing gradient problem
# Whereas, in Squeeze block global information of the feature map is extracted through its spatial dimension
then provided as input to the Excitation block. This block extract the channel wise information and corelation
between them. Therefore emphasizing on most relevant features and suppress
less important one.

![](crypto/deepresunet%20(2)%20(1).png)

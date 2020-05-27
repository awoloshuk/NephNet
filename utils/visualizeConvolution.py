'''
NOT RUNNABLE CODE. simply an example of how to generate the convolutions at each layer in the model 
'''


model_convs = nn.Sequential(*list(model.children())[0], *list(model.children())[1],
                           *list(model.children())[2], *list(model.children())[3])
#print(model_convs)
model_conv1 = nn.Sequential(*list(model_convs.children())[0:6])
model_conv2 = nn.Sequential(*list(model_convs.children())[0:13])
model_conv3 = nn.Sequential(*list(model_convs.children())[0:20])
model_conv4 = nn.Sequential(*list(model_convs.children()))
print(model_conv3)

inputs, labels = next(iter(data_loader))
torchsummary.summary(model_conv4, (1,7,32,32))
print(inputs.shape)
inputs = inputs.cuda()
output1 = model_conv1(inputs)
output2 = model_conv2(inputs)
output3 = model_conv3(inputs)
output4 = model_conv4(inputs)

print(output1.cpu().shape)
print(output2.cpu().shape)
print(output3.cpu().shape)

IMG_IDX = 0
img1 = output1.cpu().detach().numpy()[IMG_IDX]
img2 = output2.cpu().detach().numpy()[IMG_IDX]
img3 = output3.cpu().detach().numpy()[IMG_IDX]
img4 = output4.cpu().detach().numpy()[IMG_IDX]

classes = ('Glomerulus', 'Proximal Tubule', 'Vasculature')
print(classes[labels[IMG_IDX]])
og_img = np.squeeze(inputs[0].cpu().detach().numpy())[3]
print(np.squeeze(inputs.cpu().detach().numpy()).shape)
#plt.imshow(np.squeeze(inputs[0].cpu().detach().numpy())[IMG_IDX], cmap = 'gray')
#plt.pause(0.1)

for i in range(3):
    '''
    Shows convolution outputs from the various layers. Note: the input image is stack 4/7. 
    The first conv is stack 4/7 (dim = 32x32x7)
    second conv: 2/3 (dim = 16x16x3)
    third and fourth conv: stack 1/1 (dim = 8x8x1 and 4x4x1)
    '''
    f = plt.figure(figsize=(20,20))
    ax1 = f.add_subplot(161)
    ax1.set_title("Input")
    ax1.imshow(np.squeeze(inputs[0].cpu().detach().numpy())[3], cmap = 'gray')
    ax2 = f.add_subplot(162)
    ax2.set_title("First conv")
    ax2.imshow(img1[i][2], cmap='gray')
    ax3 = f.add_subplot(163)
    ax3.set_title("Second conv")
    ax3.imshow(img2[i][1], cmap='gray')
    ax4 = f.add_subplot(164)
    ax4.set_title("Third conv")
    ax4.imshow(img3[i][0], cmap='gray')
    ax5 = f.add_subplot(165)
    ax5.set_title("Fourth conv")
    ax5.imshow(img4[i][0], cmap='gray')
    plt.pause(0.1)
    
plt.show()
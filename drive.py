import carla
import torch
import numpy as np
import cv2
import torchvision.transforms as transforms
from PIL import Image
import time

# ==============================================================================
# -- Model Definition (MUST BE EXACTLY THE SAME AS IN THE TRAINING SCRIPT) =====
# ==============================================================================
# It's best practice to have this in a separate file and import it,
# but for simplicity, we will copy it directly here.

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.downsample = torch.nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels))
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(identity)
        out = self.relu(out)
        return out

class ResNetSteering(torch.nn.Module):
    def __init__(self, block, layers):
        super(ResNetSteering, self).__init__()
        self.in_channels = 64
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = torch.nn.Flatten()
        self.dropout = torch.nn.Dropout(0.3)
        self.fc = torch.nn.Linear(512, 1)
    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
        return torch.nn.Sequential(*layers)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x.squeeze(1)


# ==============================================================================
# -- Global Variables and Pre-processing ---------------------------------------
# ==============================================================================

current_steering_angle = 0.0
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Define the exact same transformations as in your training script
transform = transforms.Compose([
    transforms.Resize((220, 220)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def image_callback(image, model):
    """
    Callback function that is called every time the camera receives a new image.
    This is the heart of the inference loop.
    """
    global current_steering_angle
    
    # Convert CARLA image to a PIL Image for torchvision transforms
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # Remove alpha channel (BGRA to BGR)
    pil_image = Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB)) # Convert BGR to RGB
    
    # Apply transformations and add a batch dimension
    image_tensor = transform(pil_image).unsqueeze(0).to(device)
    
    # Make a prediction
    with torch.no_grad():
        prediction = model(image_tensor)
        current_steering_angle = prediction.item()

def main():
    """Main function to run the autonomous driving."""
    client, world, vehicle, camera = None, None, None, None
    try:
        # --- Load the trained model ---
        print("Loading trained model...")
        model = ResNetSteering(ResidualBlock, [2, 2, 2, 2]).to(device)
        model.load_state_dict(torch.load('best_model.pth'))
        model.eval()  # Set the model to evaluation mode
        print("Model loaded successfully!")
        
        # --- Connect to CARLA and spawn actors ---
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()

        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        spawn_point = world.get_map().get_spawn_points()[15] # Use a different spawn point for fun
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print("Vehicle spawned.")
        
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        # Use a resolution that's easy to process
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        
        # Start the camera listening and pass the loaded model to the callback
        camera.listen(lambda image: image_callback(image, model))
        
        # --- Main Control Loop ---
        print("Starting control loop...")
        while True:
            # Set a constant throttle for simplicity
            throttle = 0.6
            
            # Create the control command with the model's prediction
            control = carla.VehicleControl(
                throttle=throttle,
                steer=current_steering_angle
            )
            
            # Apply the control to the vehicle
            vehicle.apply_control(control)
            
            # Add a small delay to keep the loop from running too fast
            time.sleep(0.05)
            
    finally:
        # --- Cleanup ---
        print("\nCleaning up actors...")
        if camera: camera.destroy()
        if vehicle: vehicle.destroy()
        print("Cleanup complete.")

if __name__ == '__main__':
    main()
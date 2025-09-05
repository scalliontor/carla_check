import carla
import torch
import numpy as np
import cv2
import torchvision.transforms as transforms
from PIL import Image
import queue

# ==============================================================================
# -- Model Definition (MUST BE EXACTLY THE SAME AS IN THE TRAINING SCRIPT) =====
# ==============================================================================
class ResidualBlock(torch.nn.Module):
    # ... (Copy the class definition exactly from your training script) ...
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
    # ... (Copy the class definition exactly from your training script) ...
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
# -- Image Pre-processing ------------------------------------------------------
# ==============================================================================
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.Resize((220, 220)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def process_image(image, model):
    """Processes a CARLA image and returns a steering angle prediction."""
    # Convert CARLA image to PIL Image
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    pil_image = Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))
    
    # Apply transformations
    image_tensor = transform(pil_image).unsqueeze(0).to(device)
    
    # Predict steering angle
    with torch.no_grad():
        prediction = model(image_tensor)
        steering_angle = prediction.item()
    
    return steering_angle


def main():
    """Main function for synchronous autonomous driving."""
    client, world, vehicle, camera = None, None, None, None
    original_settings = None
    
    try:
        # --- Load Model ---
        print("Loading trained model...")
        model = ResNetSteering(ResidualBlock, [2, 2, 2, 2]).to(device)
        model.load_state_dict(torch.load('best_model.pth'))
        model.eval()
        print("Model loaded successfully!")
        
        # --- Connect to CARLA and enable Synchronous Mode ---
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        world.apply_settings(settings)
        
        # --- Spawn Actors ---
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        
        # Spawn at a known good location
        location = carla.Location(x=180.0, y=59.0, z=0.3)
        rotation = carla.Rotation(pitch=0.0, yaw=-90.0, roll=0.0)
        spawn_point = world.get_map().get_spawn_points()[10] 
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print("Vehicle spawned.")
        
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        
        # Create a synchronous queue to hold camera data
        image_queue = queue.Queue()
        camera.listen(image_queue.put)
        
        spectator = world.get_spectator()

        # --- Main Control Loop ---
        print("Starting synchronous control loop...")
        while True:
            # Advance the simulation and wait for the next frame
            world.tick()
            
            # Get the camera image from the queue
            image = image_queue.get()
            
            # Predict the steering angle
            steering_angle = process_image(image, model)
            
            # --- BUG FIX #2: Invert Steering Here ---
            # If the car turns left when it should turn right, uncomment the next line
            # steering_angle = -steering_angle

            # Apply control
            control = carla.VehicleControl(throttle=0.5, steer=steering_angle)
            vehicle.apply_control(control)

            # Move spectator camera to follow the car
            vehicle_transform = vehicle.get_transform()
                        # --- THE DEFINITIVE V2 Spectator Follow Logic ---
            # Get the vehicle's current transform
            transform = vehicle.get_transform()
            
            # Get the vehicle's rotation
            rotation = transform.rotation

            # Define our desired offset IN THE CAR'S LOCAL COORDINATE SYSTEM
            # x=-10 is 10m behind, z=5 is 5m up
            offset = carla.Location(x=-10.0, z=5.0)

            # The magic is here: Rotate the offset vector by the vehicle's rotation.
            # This transforms the local offset (e.g., "behind me") into a world-space direction.
            world_offset = rotation.get_forward_vector() * offset.x + rotation.get_right_vector() * offset.y + rotation.get_up_vector() * offset.z
            
            # Calculate the final world position for the spectator
            spectator_location = transform.location + world_offset
            
            # The rotation should look from the spectator towards the vehicle.
            # A simple chase-cam uses the car's yaw with a fixed downward pitch.
            spectator_rotation = carla.Rotation(pitch=-15.0, yaw=transform.rotation.yaw)
            
            # Set the new transform for the spectator
            spectator.set_transform(carla.Transform(spectator_location, spectator_rotation))
            # --- END OF DEFINITIVE V2 LOGIC ---
    finally:
        # --- Cleanup ---
        print("\nCleaning up actors and restoring settings...")
        if original_settings:
            world.apply_settings(original_settings)
        if camera: camera.destroy()
        if vehicle: vehicle.destroy()
        print("Cleanup complete.")

if __name__ == '__main__':
    main()

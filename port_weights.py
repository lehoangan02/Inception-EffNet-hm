import torch

def port_weights(input_path, output_path):
    print(f"Loading weights from {input_path}...")
    checkpoint = torch.load(input_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    
    # Extract the DOTA 'ship' class (index 6) weights for the hm head
    hm_weight = state_dict['hm.2.weight']
    hm_bias = state_dict['hm.2.bias']
    
    print(f"Original hm.2.weight shape: {hm_weight.shape}")
    print(f"Original hm.2.bias shape: {hm_bias.shape}")
    
    # Ship is index 6 in DOTA
    ship_index = 6
    state_dict['hm.2.weight'] = hm_weight[ship_index:ship_index+1].clone()
    state_dict['hm.2.bias'] = hm_bias[ship_index:ship_index+1].clone()
    
    print(f"New hm.2.weight shape: {state_dict['hm.2.weight'].shape}")
    print(f"New hm.2.bias shape: {state_dict['hm.2.bias'].shape}")
    
    checkpoint['model_state_dict'] = state_dict
    torch.save(checkpoint, output_path)
    print(f"Successfully saved ported weights to {output_path}")

if __name__ == '__main__':
    port_weights('weights_dota/model_55.pth', 'weights_dota/model_55_hrsc.pth')

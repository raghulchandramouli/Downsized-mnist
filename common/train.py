fr

def get_model_args(as_dict=False):
    
    """
    Get Model arguments from command Line or Config file. 
    If as_dict is True, return as a dictionary.
    """

    arg_dict = {
        'input_size': 40,
        'output_size': 10,
        'hidden_size': 256,
        'learning_rate' : 1e-2,
        'weight_decay' : 0,
        'batch_size' : 100,
        'total_steps' : 8000,
        'print_every' : 1000,
        'eval_every' : 250,
        'checkpoint_every' : 1000,
        'device' : 'cpu',
        'seed' : 42
    }

    return arg_dict if as_dict else ObjectView(arg_dict)
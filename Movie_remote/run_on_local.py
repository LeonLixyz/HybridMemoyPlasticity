import subprocess

commands = [    
    'Nonl_RO [100,100] [1] [M] 4',
    'Nonl_RO [100,100] [1] [A] 4',
    'ML [100] [1] [M] 4',    
    'ML [100] [1] [A] 4',    
    'ML [50,50] [1,1] [M,A] 4',    
    'ML [50,50] [1,1] [M,M] 4',    
    'Dyn_RO [50,50] [1,1] [M,A] 4',    
    'Dyn_RO [50,50] [1,1] [M,M] 4',    
    'Stack [50,50] [1,1] [M,A] 4', 
    'Nonl_RO [50,50] [1] [M] 4',
    'Nonl_RO [50,50] [1] [A] 4',
    'ML [50] [1] [M] 4',    
    'ML [50] [1] [A] 4',    
    'ML [25,25] [1,1] [M,A] 4',    
    'ML [25,25] [1,1] [M,M] 4',    
    'Dyn_RO [25,25] [1,1] [M,A] 4',    
    'Dyn_RO [25,25] [1,1] [M,M] 4',    
    'Stack [25,25] [1,1] [M,A] 4',   
    'end',
    ]

for command in commands:
    print(command.split())
    subprocess.run(['python', 'main.py'] + command.split())
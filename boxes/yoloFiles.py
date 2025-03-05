def loadFile(videoName):
    # Extract filename from path
    videoName = videoName.split('/')
    videoName = videoName[len(videoName) - 1]
    
    # Construct path to boxes file
    videoName = 'boxes/' + videoName.split('.')[0] + '.txt'
    
    # Read and parse file
    with open(videoName, "r") as f:
        lines = f.readlines()
    
    lines = [x.strip() for x in lines]
    lines.pop(0)  # Remove header line

    temp = []
    res = []

    # Parse detection boxes
    for l in lines:
        if l == '--':
            res.append(temp)
            temp = []
            continue
            
        x = l.split()
        
        # Handle multi-word class names
        if x[0] in ['traffic', 'fire', 'stop', 'parking', 'sports',
                    'baseball', 'tennis', 'wine', 'hot', 'cell',
                    'teddy', 'hair']:
            x[0] = x[0] + x[1]
            x.pop(1)

        # Format: [class, left, right, top, bottom, confidence]
        temp.append([x[0], float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])])
        
    # Add final batch if not empty
    if temp:
        res.append(temp)
        
    return res
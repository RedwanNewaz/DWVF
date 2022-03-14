import numpy as np




def sampling_locations(num_samples, area, sample_dist, offset=2, seed=5):
    '''
    :param num_samples: number of point instances in the scene = num of obs + goal + init pos
    :param area: 2D target area, must be positive number R+
    :param sample_dist: required distance between each instance
    :param offset: enforce placing instance inside while avoiding boundaries
    :param seed: for numpy random number seed
    :return: a list of 2D coords
    '''
    np.random.seed(seed)
    samples = []

    bound = lambda x: max(offset, min(x - offset, x))
    area = list(map(bound, area))


    while len(samples) < num_samples:

        obs = np.random.uniform(area[0], area[1], size=(1, 2))

        if len(samples):
            dist = min(map(lambda x: np.linalg.norm(obs - x), samples))
            if dist > sample_dist:
                samples.append(obs)
        else:
            samples.append(obs)
    result = np.reshape(samples, (num_samples, 2))
    np.random.shuffle(result)
    print(f"offset = {offset}, bound area = {area}, num instances = {len(result)}")

    return result

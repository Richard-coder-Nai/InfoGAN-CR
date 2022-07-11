from __future__ import division
import os
import numpy as np


def load_3Dshapes(path):
    imgs = np.load(os.path.join(path,'images.npy'), mmap_mode='r')
    latent_values = np.load(os.path.join(path, 'labels.npy'))

    latents_names = ('floor color', 'wall color', 'object color', 'object size', 'object type', 'azimuth')
    latents_sizes = np.array([10, 10, 10, 8, 4, 15])
    latents_possible_values = {
            'floor color' : np.arange(10), 
            'wall color' : np.arange(10), 
            'object color' : np.arange(10), 
            'object size' : np.arange(8), 
            'object type' : np.arange(4), 
            'azimuth' : np.arange(15), 
            } 
    latents_bases = np.concatenate(
        (latents_sizes[::-1].cumprod()[::-1][1:], np.array([1, ])))

    def latent_to_index(latents):
        return np.dot(latents, latents_bases).astype(int)

    def sample_latent(size=1):
        samples = np.zeros((size, latents_sizes.size))
        for lat_i, lat_size in enumerate(latents_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=size)
        return samples

    metric_data_groups = []
    L = 100
    M = 500

    for i in range(M):
        fixed_latent_id = i % 6
        latents_sampled = sample_latent(size=L)
        latents_sampled[:, fixed_latent_id] = \
            np.random.randint(latents_sizes[fixed_latent_id], size=1)
        # print(latents_sampled[0:10])
        indices_sampled = latent_to_index(latents_sampled)
        imgs_sampled = imgs[indices_sampled]
        metric_data_groups.append(
            {"img": imgs_sampled,
             "label": fixed_latent_id - 1})

    selected_ids = np.random.permutation(range(imgs.shape[0]))
    selected_ids = selected_ids[0: imgs.shape[0] // 10]
    metric_data_eval_std = imgs[selected_ids]

    random_latent_ids = sample_latent(size=imgs.shape[0] // 10)
    random_latent_ids = random_latent_ids.astype(np.int32)
    random_ids = latent_to_index(random_latent_ids)
    assert random_latent_ids.shape == (imgs.shape[0] // 10, 6)
    random_imgs = imgs[random_ids]

    random_latents = np.zeros((random_imgs.shape[0], 6))
    for i in range(6):
        random_latents[:, i] = \
            latents_possible_values[latents_names[i]][random_latent_ids[:, i]]


    metric_data_img_with_latent = {
        "img": random_imgs,
        "latent": random_latents,
        "latent_id": random_latent_ids,
        "is_continuous": [False, False, False, False, False, False]}

    metric_data = {
        "groups": metric_data_groups,
        "img_eval_std": metric_data_eval_std,
        "img_with_latent": metric_data_img_with_latent}

    return imgs, metric_data, latent_values


if __name__ == "__main__":
    load_3Dshapes("../data/3dshapes")

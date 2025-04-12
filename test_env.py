
import argparse
#from carla_game.carla_benchmark import CarlaBenchmark as CarlaEnv
from carla_game.carla_gamev09 import CarlaEnv
from common.utills import print_square
from model.test_agents import mathDriver, randomDriver, crashDriver
from PIL import Image
import numpy as np
'''

        "ClearNoon",
        "WetCloudyNoon",
        "MidRainyNoon",
        "ClearSunset",
        "WetCloudySunset",
        "MidRainSunset"

'''

def get_sample_image():
    city_names  = [
        'Offroad_1',
        'Offroad_2',
        'Offroad_3',
        'Offroad_4',
        'Offroad_5',
        'Offroad_6',
        'Offroad_7',
        'Offroad_8'
    ]


    weathers = [
        "ClearNoon",
        "WetCloudyNoon",
        "MidRainyNoon",
        "ClearSunset",
        "WetCloudySunset",
        "MidRainSunset"
    ]

    for city_name in city_names:
        for i, weather in enumerate(weathers):
            env = CarlaEnv(
                log_dir='./CarlaLog.txt',
                render=False,
                plot=False,
                server_size=(4, 3),
                city_name=city_name,
                weather=weather,
                is_image_state=False
            )

            if i==0:
                init_pos = env.init_pos
                init_state = env.init_state
            else:
                env.init_pos = init_pos
                env.init_state = init_state

            state = env.reset(reset_position=False)


def main(args):
    '''make environment'''
    env = CarlaEnv(
        log_dir='./CarlaLog.txt',
        render=args.render,
        render_gcam=False,
        plot=args.plot,
        server_size=(4, 3),
        city_name=args.city_name,
        weather=args.weather,
        is_image_state=False
    )

    #agent = crashDriver()
    agent = mathDriver(speed_limits = [5, 6])

    '''verbose'''

    verbose = {}
    for key in vars(args):
        verbose[key] = getattr(args, key)
    print_square(verbose)

    '''main loop'''

    total_step = 0
    total_reward = 0
    try:
        for episode in range(args.max_episode):
            state = env.reset()
            episode_reward = 0
            step = 0
            done = False
            while not done:
                step += 1

                if args.render:
                    if episode % 100 == 0:
                        # save the rendered images
                        env.render(save=False, step=step, model="None")
                    else:
                        env.render()

                action = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)

                rgb_image, _, _ = env.get_full_observation()
                if rgb_image.ndim == 3 and rgb_image.shape[0] == 1:  
                    rgb_image = np.squeeze(rgb_image, axis=0)  
                if rgb_image.shape[0] == 3:
                    rgb_image = np.transpose(rgb_image, (1, 2, 0))
                # print(rgb_image.shape)
                # print(rgb_image.min(), rgb_image.max(), rgb_image.dtype)

                rgb_image = (rgb_image * 255).clip(0, 255).astype(np.uint8)
                image = Image.fromarray(rgb_image)
                image.save(f"./rgb/output_image_step_{step}.png")

                next_obs = next_state
                total_step += 1
                episode_reward += reward
                state = next_state

                print_square({
                    "reward": reward,
                    "steering": action[1],
                    "throttle": action[0],
                    "speed": env.epinfos["speed"],
                    "track_width": env.epinfos["track_width"],
                    "mileage": env.epinfos["mileage"],
                    "distance_from_center": env.epinfos["distance_from_center"],
                    "current_step": env.epinfos["current_step"],
                    "is_collision": env.epinfos["is_collision"],
                })

                if total_step > args.max_step:
                    break

            total_reward += episode_reward
            print("#{}-th Episode, Total_Reward {}".format(episode, episode_reward))

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    finally:
        env.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--city-name', default='Offroad_2', type=str)
    parser.add_argument('--weather', default='ClearNoon', type=str)
    parser.add_argument('--max-episode', default=1, type=int)
    parser.add_argument('--max-step', default=100, type=int)
    args = parser.parse_args()
    args.render = True
    args.plot = False
    main(args)

    #get_sample_image()


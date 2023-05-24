# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 01:42:31 2023

@author: napat
"""

        with torch.no_grad():
            next_actions, next_log_probs, _ = agent.get_action_prob(batch_nextstates)
            next_q1 = agent.target_critic1(torch.cat((batch_nextstates, next_actions), dim=1))
            next_q2 = agent.target_critic2(torch.cat((batch_nextstates, next_actions), dim=1))
            min_next_q = torch.min(next_q1, next_q2)
            soft_state = min_next_q - agent.alpha * next_log_probs
            target_q = batch_rewards + (1 - batch_done) * agent.gamma * soft_state
            
        pred_q1 = agent.critic1(torch.cat((batch_states, batch_actions), dim=1))
        pred_q2 = agent.critic2(torch.cat((batch_states, batch_actions), dim=1))

        loss1 = agent.criterion(pred_q1, target_q)
        loss2 = agent.criterion(pred_q2, target_q)

        n_episode=10
        delay=0.1
        rewards = []
        steps = []
        for i in range(n_episode):
            state, _ = agent.env.reset()
            done, truncated = False, False
            eps_reward = 0
            n_steps = 0
            while not (done or truncated):
                action = agent._choose_action(state)
                nextstate, reward, done, truncated, _ = agent.env.step(action)
                print(reward)
                state = nextstate
                agent.env.render()
                time.sleep(delay)
                # print(action)
                eps_reward += reward
                n_steps += 1
                
            rewards.append(eps_reward)
            steps.append(n_steps)
            print(f'Episode {i}/{n_episode} \t Reward: {eps_reward:.4f} \t Length: {n_steps}')
        rewards = np.array(rewards)
        steps = np.array(steps)
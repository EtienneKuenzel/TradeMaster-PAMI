# PAMI: Data Augmentation for Financial Reinforcement Learning
-This is a Repository of the Pattern-Analysis  and Machine-Intelligence Course(Summer 2024)<br/>
-We used [TradeMaster](https://github.com/TradeMaster-NTU/TradeMaster) as our Simulator

## Problems with TradeMaster ##
While Trademaster promises a range of capabilities, it suffers from several significant and minor issues.
First, the installation process for Trademaster is problematic, with missing dependencies that render
the RL environment non-functional when using the provided requirements.txt to create a Conda envi-
ronment. Some package versions were incorrect, others were missing, and certain packages needed to
be installed via mim rather than pip.<br/>
Second, after successfully setting up Trademaster on our local machine, we ran multiple tutorials pro-
vided in both GitHub and a Jupyter Notebook. Of the six pre-made tutorials, only three executed
correctly, while the remaining three failed, generating various errors such as NotImplementedError and
KeyError. Leaving us with just the DQN implemented for algorithmic<br/>
Third, there is a lack of comprehensive guidelines for using the platform. While the tutorials exist,
they do not explain how to combine different algorithms effectively, leaving users to rely on trial and
error due to the environment’s obscure and unintuitive nature. Furthermore, some known issues have
been left unresolved, despite being reported as early as last year.<br/>
Lastly, the code itself is inconsistent in both documentation and structure. While comments are
mostly in English, some are in Chinese, and certain environments are implemented multiple times
under different names with no apparent differences. Additionally, the code contains typos and spelling
errors
## Abstract ##
Investing is the process of allocating money or resources into assets or projects to generate a return or
profit over time. One of the main ways to invest today is to buy stocks, real estate, cryptocurrencies,
and other types of assets. While the overall market has been going up over the last few years making
long-term investing a good choice, outperforming this market over many years with short-term trading
while still having low volatility is an achievement that only a few professionals achieve.<br/>
The hard-to-learn patterns that emerge in stock and cryptocurrency trading make it a good problem to
utilize and test machine learning approaches. This field is called algorithmic trading and consists of an
agent trying to buy an asset before it goes up and sell it before it goes down to maximize returns. In
this context, reinforcement learning (RL) has gained traction as a powerful tool for developing adaptive
trading algorithms capable of learning and evolving from market interactions. By continuously refining
strategies through iterative feedback, RL methods have the potential to outperform traditional models
in dynamic and unpredictable market conditions.<br/>
With RL profiting from curricula in other problems(a method that structures the training process to
gradually introduce more complex scenarios) we will research the impact of different curricula in this
domain and if they improve certain aspects and capabilities of our base model.<br/>
A crucial aspect of training effective RL models is the availability of high-quality data. The creation
of diverse and representative training datasets is essential for robust model performance. We explore
the use of diffusion techniques to generate synthetic training data, thereby expanding the dataset and
ideally providing RL models with a more comprehensive understanding of market behaviors.<br/>
This project in the PAMI course aims to investigate the synergy between reinforcement learning,
curriculum learning, and data augmentation through diffusion, focusing on their collective impact on
enhancing the training process of reinforcement learning agents tested on historical Bitcoin data. By
integrating these advanced techniques, we seek to contribute to the development of more effective and
adaptive trading systems capable of navigating the complexities of the digital asset landscape.

本次 2023 年 3.24-3.25 於[桃園大溪笠復威斯汀酒店](https://goo.gl/maps/e6Bin12WoKCCTgTh9)所舉辦的 [5th Augmented Intelligence and Interaction (AII) Workshop](https://elsa-lab.github.io/AIIWorkshop_5th/) 邀集台灣 ML/DL/AI 相關研究的頂尖研究者 (估計~90%相關領域的台灣知名學者) ，來分享他們近期的研究成果。 其中更有兩位來自美國的華人學者: [李飞飞](https://profiles.stanford.edu/fei-fei-li) 與 [楊明玄](https://scholar.google.com/citations?user=p9-ohHsAAAAJ&hl=zh-TW)，作為 Keynote ，為本次活動的一大看點。

我是 Kuan，在這裡我分享本次 Workshop 我個人整理的內容。 如果大家覺得實用歡迎分享，也歡迎提供任何補充或 fork 本筆記。 Disclaimers: 當天的演講很多，我的注意力有限，有部分講者的內容紀錄的比較少

這裡是我的 LinkedIn profile: ([Kuan-Chun Lee](https://www.linkedin.com/in/kuan-chun-l-a7745560/))，如果大家對於本筆記有任何疑問或建議歡迎與我聯繫!

<!-- /個人 side notes (不顯示) -->
<!-- 第一次筆記: 講者 與它們談論的架構 -->
<!-- 第二次筆記: 講者談論的具體內容 -->
<!-- 第三次筆記: 一些自己的想法或是其他的 + 一些 ChatGPT 的建議 -->
<!-- ** 部分我們可以讓 chatGPT 回答一下 -->

-------------------------------------

# 筆記內容
## Day 1 (2023.03.24)
### 1 [李飞飞](https://profiles.stanford.edu/fei-fei-li)

- 講題: **From Seeing to Doing: Understanding and Interacting with the Real World.**
- 演講架構:
    - 研究的歷史脈絡
    - Seeing is for Understanding        
    - Seeing is for Understanding: Advanced        
    - Seeing is for Doing        
    - Robotics             
    - Conclusion   
- 演講內容:
    - 研究的歷史脈絡
        - 寒武紀大爆發: 遠古海洋生物在產生視覺以後，考古學家認為是生物智能演化的一大里程碑，開啟了地球演化史的新章節。
        - > vision is the corner stone of intelligence
        - 分享幾個早期的 visual perception 相關心理學與神經科學研究 (60-90年代)
    - Seeing is for Understanding
        - 她的博士期工作: 應用 RSVP 實驗法，發現人類完整視覺感知 (?關於 visual categorization) 的尺度約 500ms 且在 30-70ms 即可達到充分的理解
        - 早期的 ojbect recognition 的研究與限制
        - J.J. Gibson 的 ecological approach**: 
        > (we shall study) not what is inside your head but what your head is inside of.
        - 更 ecological 的 object recognition task: [ImageNet (CVPR2009)](https://www.image-net.org/about.php), ~2200 label types & 15M datapoints
    - Seeing is for Understanding: Advanced
        - 在 ImageNet 之後，思考如何在 object recognition 之上，更進一步理解視覺。
        - Jeremy Wolfe 的 object relationship 思路:
        > ...One could imagine a set of systems for object recognition in which each system was optimized for a particular viewpoint or lighting condition or background, and then the problem of recognition would simply be the problem of picking the system that was best suited to the current object relationship..."
        - 從以上思路出發它們實驗室發展了兩個 Benchmark 資料集:
        - Visual Genome Project:
            - [IJCV 2017](https://arxiv.org/abs/1602.07332): 每個 image 上面有以關係圖表式的 object relationship 標註
            - [ECCV 2016](https://arxiv.org/abs/1608.00187): 訓練出對於 object relationship 進行解釋的 model 並且有能力解釋 long-tail cases 例如 a horse wears a cap.
        - Action Genome Project: 
            - [CVPR 2020: Action Genome: Actions as Compositions of Spatio-temporal Scene Graphs](https://www.actiongenome.org/): 影片理解的 benchmark 資料集，延伸 scene graph 關係圖標註方式到影片上
            - Multi-Object Multi-Activity understanding: 
                - 動態的 object activity 理解
                - [Neurips 2022: MOMA-LRG](https://proceedings.neurips.cc/paper_files/paper/2022/hash/22c16986b2f50af520f56dc34d91e403-Abstract-Datasets_and_Benchmarks.html)
                - [NeurIPS 2021: MOMA](https://proceedings.neurips.cc/paper/2021/hash/95688ba636a4720a85b3634acfec8cdd-Abstract.html)
        - 其他關於 visual understanding 的工作: 一些早期的 image captioning 工作
    - Seeing is for Doing
        - 研究背景 
            - 柏拉圖的 Allegory of the Cave 比喻
            - [\<Other Minds\>](https://www.amazon.com/Other-Minds-Octopus-Origins-Consciousness/dp/0374227764) 一書中提到的
            > the ... fundamental funtion of the nervous system is to link perception to action
            - [Held and Hein 1963: Kittens Carousel 實驗](http://embodiedknowledge.blogspot.com/2011/12/classic-experiment-by-held-and-hein.html): perceptual learning required guided interaction
            - 鏡像神經元: 透過觀察他人的行為來學習如何行為
        - 實驗室關於 Seeing is for Doing 的相關工作:
            - about learning algorithms, represetnation learning, action & planning
            - 有一張整個實驗室工作方向的摘要圖(*參考下方參考資料)
        - Learning to explore:
            - Learning to Play (2005), 在她早期的實驗工作中，觀察到嬰兒在隨意遊玩的過程中，產生有結構的行為
            - [NeurIPS 2018](https://arxiv.org/abs/1802.07442) 算法重現 learning to play 現象。通過 world model-based intrinsic movitation，Agent 可以自主發展出合理的行為
    - Robotics
        - 研究背景
            - 認為目前 Robotics/RL 相關的 benchmark 都太過著重於 skill-level 和 short-horizon goal 的任務，且過於簡化。無法實際解決真實世界中的 Robotics 問題
        
        - 幾個相關算法研究
            - [ICRA 2018: Neural Task Programming](https://arxiv.org/abs/1710.01813): , 嘗試解決一類 long-horizon task (object sorting)
            - [RSS 2021: Learning Generalizable Skills via Automated Generation of Diverse Tasks](https://sites.google.com/view/rss-slide/): 使用生成模型去生成 robotic learning 的 subgoals/curriculum
            - [ICLR 2021: APT-GEN](https://arxiv.org/abs/2007.00350): learning task 的生成

        - BEHAVIOR Benchmark
            - 在一次從 JJ Gibson 的思路出發，微改寫他的 ecological approach:
            > ... where your head & body is inside of
            - 有一系列工作，鋪陳到現在最新的 BEHAVIOR Benchmark
                - [iGibson](https://ieeexplore.ieee.org/abstract/document/9636667)                
                - [BEHAVIOR](https://behavior.stanford.edu/)
                - [BEHAVIOR-1k](https://proceedings.mlr.press/v205/li23a.html), aka [OmniGibson](https://github.com/StanfordVL/OmniGibson)
            - BEHAVIOR Benchmark介紹:
                - 內容: 居家任務的機器人 3D 動態學習環境 
                - 目的是開發機器人輔助人類，因此主要提供人類最希望請機器人幫忙的相關任務，提供機器人學習。 根據統計，人類最希望被幫忙的家務主要跟清掃整理的勞力活有關。
                - 使用 BDDL 語言定義任務
                - 目前的 state-of-the-art 算法仍無法解決此 Benchmark 中的任務，接近 0 的表現值。
            
    - Conclusion
        - "Vision is the cornerstone of intelligence"
        - Gibson's ecological approach 貫串了整個她的研究方法以及所設計的 Benchmark 資料集

- 後記: 其他相關紀錄
    - 飞飞 @ Panel Discussion:
        - 回答 Mediatek 梁柏嵩: 台灣能從 NAAR (?名字可能是錯的) 學到甚麼經驗? 這是在美國一個關於提供 open data, computing resouces 給學生以及 public sectors 的大型計畫。
        - 問李宏毅: 當我們完成了能跟人自然對話的 "Speech ChatGPT" 之後，你有沒有想像過這個世界會有甚麼變化?
            - 李宏毅: 目前有一個想法是通過和 Speeech ChatGPT 練習對話，也許能夠幫助講話比較沒信心的人(例如肥宅)，重新找回講話的信心。
        - 其他的一些想法分享: 希望 AI 研究者的人不要把自己只定位在 application layer 也許還能更深入地去了解 intelligence 的基本問題；鼓勵 AI 人往非傳統 ee/cs 的 multi-disciplinary 發展，比如 material science, history, cancer study (cancer clinician), education, 等其他各種可能的新領域。
    
    - Benchmark 資料集整理:
        - [2009: ImageNet](https://www.image-net.org/)
        - [2016: Visual Genmoe (scene graphs)](https://arxiv.org/abs/1602.07332)
        - [2020: Action Genome (video/dynamic scene graphs)](https://www.actiongenome.org/)
        - [2021: BEHAVIOR](https://behavior.stanford.edu/)
        - [2023: BEHAVIOR-1k](https://proceedings.mlr.press/v205/li23a.html)

    - 參考資料:
        - Andrew Parker (2004): [\<In The Blink Of An Eye: How Vision Sparked The Big Bang Of Evolution\>](https://www.amazon.com/Blink-Eye-Vision-Sparked-Evolution/dp/0465054382)
        - James J Gibson (1979): [The Ecological Approach to Visual Perception](https://www.amazon.com/Ecological-Approach-Perception-Psychology-Routledge/dp/1848725787)
        - Jeremy Wolfe (1994): [Guided Search 2.0 A revised model of visual search](https://link.springer.com/article/10.3758/BF03200774)
        - Peter Godfrey-Smith: [Other Minds](https://www.amazon.com/Other-Minds-Octopus-Origins-Consciousness/dp/0374227764) [飞飞推薦閱讀]
        - 演講的簡報
            - 雖然本次沒有直接拿到獎者的簡報，不過近期她有幾個 talks 內容重疊蠻多，可以參考:
                - [Scale-AI talk](https://www.youtube.com/watch?v=XzFyYXovHMU), [Lab 研究方向摘要位置 19:49](https://youtu.be/XzFyYXovHMU?t=1189)                
                - [talk @ AMLC 2022](https://www.youtube.com/watch?v=rrrV-cP4wnw)
                

### 2 [徐宏民](https://winstonhsu.info/)
- 講題: 3D Comprehension for Efficient Robotic Manipulation
- 演講內容:
    - 筆記較少
    - 主要分享佈署 robotic learning (主要是抓取(grasping)任務) 相關問題的一些經驗分享
    - 其中有提到若干篇相關 paper

### 3 [王鈺強](http://vllab.ee.ntu.edu.tw/ycwang.html)
- 講題: Feature Pyramid Diffusion for Complex Scene Image Synthesis
- 演講內容: 
    - 分享 NVIDIA 相關內容    
        - 介紹近期 text2image models 的演進，提到 NVIDIA 的 counterpart, Edify
        - 引用了一篇 Sequotiacap (紅杉資本) 關於 Generative AI 到 2030 年的發展預測 [Generative AI: A Creative New World](https://www.sequoiacap.com/article/generative-ai-a-creative-new-world/) (這並發演講重點，只是我個人蠻喜歡這個分析裡面的圖表)

    - 分享兩篇近期的研究
        - [NeurIPS 2022: Paraphrasing Is All You Need for Novel Object Captioning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/2a8e6c09a1fd747e43a74710c79efdd5-Abstract-Conference.html): 
            - 任務: image captioning in the wild (需要 label unknow objects)
            - 使用 LLM 解決 unknown class 的問題
        - [AAAI 2023: Frido: Feature Pyramid Diffusion for Complex Scene Image Synthesis](https://arxiv.org/abs/2208.13753): Manipulation of Image Synthesis
            - 任務: manipulation of image synthesis
            - 定義兩個向度(fidelity & ?), 應用 diffusion model 設計 coarse level & fine level 內容生成的控制流程

### 4 [陳佩君](https://scholar.google.com/citations?user=zUi7TfgAAAAJ&hl=zh-TW)
- 講題: On Breakthroughs of Generative AI towards AGI
- 演講內容
    - 筆記較少
    - 分享 Microsoft Generative AI 相關產品/研究分享: 
        - 365 copilot 
        - bing.com/create: bing 的 text2image 服務
        - bing.com/new: bing 的 GPT-powered engine
        - [Sparks of Artificial General Intelligence: Early experiments with GPT-4 (2023.03)](https://arxiv.org/abs/2303.12712): Microsoft/OpenAI 對 GPT-4 進行的一些實驗

### 5 [陳維超](https://scholar.google.com/citations?user=bndb0gYAAAAJ&hl=en)
- 講題: The Right Problem to Solve in Manufacturing
- 演講內容
    - 筆記較少
    - 分享在英業達解決生產(Manufacturing)問題的經驗        
        - 有設計一個有趣且有效的工作流程:
            - simplicity & impact -> adaptive goals -> data governance (?data 端方面的議題，沒記清楚) -> trustworthy and transferrable

### 6 [孫民](https://scholar.google.com/citations?hl=zh-TW&user=1Rf6sGcAAAAJ&view_op=list_works&sortby=pubdate)
- 講題: AI for 3D Indoor Space
    - movitation: 90% robotic tasks are indoor
- 演講內容
    - 介紹兩篇近期的兩個研究
    - [NeuRIPS 2022: 360-MLC: Multi-view Layout Consistency for Self-training and Hyper-parameter Tuning](https://arxiv.org/abs/2210.12935)
        - 使用 multi-view self-supervised learning 學習 3D indoor layout projection
        - 設計 entropy-based metric 去?
    - 3D Object Detection (collaboration with Amazon)
        - 相關的 reference 沒有記到


### 7 [吳毅成](https://cgilab.nctu.edu.tw/~icwu/ch_index.html)
- 講題: A Novel Approach to Solving Goal-Achieving Problems for Board Games
- 筆記較少
- 演講內容:
    - 介紹一些改進 AlphaZero (應用到圍棋)的相關工作
    - [ICLR-2022: ](https://openreview.net/forum?id=nKWjE4QF1hB)  
        - 提到 "RZS" 算法
        - 減低搜尋速度，提升贏取遊戲的時間(相較原本的純粹提升勝率)
        - 未來可能應用到 goal-achieving prolems, e.g. Theorem proving

### 8 [賴尚宏](http://www.cs.nthu.edu.tw/~lai/)
- 講題: RGB-D Face Recognition with Identity-Style Disentanglement and Depth Augmentation
- 演講內容:
    - 主要介紹一篇近期的 paper: [IEEE BBIS 2023: RGB-D Face Recognition with Identity-Style Disentanglement and Depth Augmentation](https://ieeexplore.ieee.org/abstract/document/10011574/keywords#keywords)
    - 工作目標: 設計一個 Robust 的 RF 系統
    - 想法: 使用 RGB + Depth (RGB+D) 當作 image features
    - 挑戰1: 市面上缺乏 RGB+D 的大型開源數據集
        - 資料數量: 2D vs 3D ~ O(10M) v O(10K)
    - 挑戰2: 如何有效的利用 RGB+D 的特徵做預測
    - 解決方法:
        - 使用/自行訓練 depthNet 對 2D dataset 做 depth 維度的預測，以達到資料擴增 (data augmentation)
        - 設計 disentanglement encouraging self-supervised learning objective 目標解決挑戰二
    
        
### 9 [陳煥宗](https://htchen.github.io/)
- 講題: Multiview Regernerative Morphing
- 演講內容:
    - 主要分享一篇近期的 paper: [ECCV 2022: Multiview Regenerative Morphing with Dual Flows](https://arxiv.org/abs/2208.01287)
    - 幾個關鍵詞:
        - image morphing: 把影像中的 object 進行一些幾何上扭轉
        - multi-view: 對某個 object 的不同視角 (通常從3D的角度去看)
        - regenerative morphing: 把某一類的 (paper 裡面的) object 透過平滑的形變過程轉成另外一類 (例如 paper 裡面 figure 10 把熱狗轉成一盆植物)。 想法來自於 [CVPR 2010: Regenerative Morphing](https://grail.cs.washington.edu/projects/regenmorph/)
    - 提出的解決方案:
        - 主要使用到兩個技術想法
            - [NeRF](https://arxiv.org/abs/2003.08934): Neural Radiance Field
            - Optimal Transport 
        - 主要流程參考 paper 中的 Fig. 2
        - 也應用實驗室開發的 DVGO 算法
            - 功能: (multi-) view synthesis from images; 從某張圖片生成該圖片的視角
            - DVGO: [github](https://github.com/sunset1995/DirectVoxGO), papers: [DVGO](https://arxiv.org/abs/2111.11215), [DVGO-v2](https://arxiv.org/abs/2206.05085)
            

### 10 [王傑智](https://sites.google.com/site/chiehchihbobwang/home?authuser=0)
- 講題: Lessons Learned from Self-Driving Car Operations on Public Roads in Taiwan and Australia
- 演講內容
    - 筆記較少
    - 分享一些自動駕駛相關實務經驗以及佚事的分享
        - 2022 年在澳洲墨爾本測試自架卡車
            - 兩個工程重點技巧:
                - 1 提高 sensor redundancy 以確保安全性
                - 2 整合至當地的交通系統 (例如 camera 等 sensor )
        - 在桃園國際機場的測試 Demo

### 11 [林守德](https://www.csie.ntu.edu.tw/~sdlin/)
- 講題: Environment Diversification with Multi-Head Neural Networks for Invariant Learning
- 演講內容:
    - 介紹一篇近期的 paper: [NeurIPS 2022: 同講題](https://openreview.net/forum?id=FDmIo6o09H)
        - 工作目標: Unsupervised Out-of-Distribution (OOD) Generalization
        - 應用 invariant learning 的想法 (features 切開成 variant/invariant features 再對 variant features 進行一些額外的處理)，設計出 a set of 3 losses 
        - 預期的 benefits of work: 1. de-biasing & 2. prompting fairness of ML algorithms

### 12 [邱維辰](https://walonchiu.github.io/)
- 講題: On the Long Way to Learning Depth and Dynamic Perception
- 演講內容:
    - 講者討論了他近幾年關於 real-world (3D) vision 的一系列研究
    - real-world visual perception 有兩要件:
        - 3D depth perception (depth estimation)
        - environment dynamics perception (optical flow)
    - 研究工作:
        1. 研究問題1: 如何使用非 supervised 的方法 learn 3D perception
            - [2019 CVPR: Bridging Stereo Matching and Optical Flow via Spatiotemporal Correspondence](https://lelimite4444.github.io/BridgeDepthFlow-Project-Page/)            
            - 思路: 應用 multi-task unsupervised learning (self-supervised) 間接學習 (encode) 3D information
        2. 研究問題2: 如何得到便宜的 labeled data 
            - [ECCV 2022: 3D-PL: Domain Adaptive Depth Estimation with 3D-aware Pseudo-Labeling](https://arxiv.org/abs/2209.09231)
            
            - 思路: unsupervised domain adaptation, 訓練 model 在 3D 的 synthetic data 上面，在應用它到實際 data 上面做 pseudo-label prediction
            - 還有一篇 [ICCV 202x] 相關 paper 沒有記錄到
        4. [ICCV 2023 submitted] 
            - 使用 data augmentation 加上 predicted optical flow data (predicted from depth data) 來增加標註資料量
            - 目前本 paper 貌似還沒釋出
        5. 總結
            - 目前 3D perception 在技術上面還不夠成熟
            - 且資料收集仍是一個待解問題
            - multi-modal fusion 會是另外一個非常重要的方向 (相關工作: [CVPR 2023](https://arxiv.org/abs/2303.03369))


### 13 [李濬屹](http://cymaxwelllee.wixsite.com/elsa)
- 講題: Visual-to-Real: Vision Based Navigation
- 演講內容:
    - 筆記較少
    - 介紹實驗室的一系列工作以及成果 demo 
        - 核心的做法是 learning visual intermediate representation (e.g. segmantation, optical flow) for downstream control tasks
        - 幾篇討論的 paper (可能沒列到全部):
            - [IJCAI 2018: Virtual-to-Real: Learning to Control in Visual Semantic Segmentation](https://scholar.google.com/citations?view_op=view_citation&hl=zh-TW&user=5mYNdo0AAAAJ&citation_for_view=5mYNdo0AAAAJ:e_rmSamDkqQC)
            - [IROS 2022: Investigation of Factorized Optical Flows as Mid-Level Representations](https://ieeexplore.ieee.org/abstract/document/9981638)
                - OF 分解成 ego-OF 以及 object-OF
                - demo 用 deja vu 背景音樂!?
            - [ECCV 2022: S2F2: Single-Stage Flow Forecasting for Future Multiple Trajectories Prediction](https://scholar.google.com/citations?view_op=view_citation&hl=zh-TW&user=5mYNdo0AAAAJ&sortby=pubdate&citation_for_view=5mYNdo0AAAAJ:a9-T7VOCCH8C)
            - [IROS 2023 submitted: Vision based Virtual Guidance for Navigation](https://arxiv.org/abs/2303.02731)


### 14 [胡敏君](http://mislab.cs.nthu.edu.tw/)
- 講題: AI in Sport
- 演講內容
    - 筆記較少
    - 介紹 AI in Sport 的相關工作以及在新創公司 [Neuinx](http://neuinx.com/) 的工作
        - CES 2023 有獲獎項


### 15 [謝秉均](https://pinghsieh.github.io/)
- 講題: Q-Pensieve: Boosting Sample Efficiency of Multi-Objective RL Through Memory Sharing of Q-Snapshots
- 演講內容:
    - 介紹一篇論文: [ICLR 2023: 同演講題目](https://arxiv.org/abs/2212.03117)
        - 本論文工作目標: 提升 MORL (multi-ojbetive RL) 的數據效率
        - 講者摘要兩個 takeway:
            > 1 One should always use MORL (even if it is single-objective RL problem)

            > 2 Policy-level knowledge sharing is the key for MORL
        - MORL 問題定義與挑戰:
            - 定義: 訓練一組 vector-value function 使得任意的 multi-objective weighting $\lambda$ 都可以達到 optimal values
            - 因此很顯然的挑戰是 weighting $\lambda$ 的組合有無限多種
    - 解決方法: 
        - 記憶使用 Q-Pensieve: 
            - "儲思盆 (Pensieve) 是一種用來查看記憶的物品。" 想法取自哈利波特 ([source](https://harrypotter.fandom.com/zh/wiki/%E5%86%A5%E6%83%B3%E7%9B%86?variant=zh-tw))
            - 引入另外一個新的 replay buffer: Q-Pensieve, 裡面儲存學期過程中對不同的 weighting $\lambda$ 進行學習的 Q-newtwork snapshots            
            - 這些 snapshots 具有對 objetive 的知識，因此作者認為能夠幫助學習。在後續實驗顯示，Q-pensieve 能夠:
                1) 提升 MORL 的 sample-efficiency (論文 table1, figure 2)
                2) 在 single-objective 學習過程中，能避免掉到 sub-optimal policy regime (論文 figure 3)
        - Q-Pensieve 整合到 SAC 的架構下面 
            - (主要貢獻) 推出一個 Q-Pensieve Soft Policy Improvement 公式 (論文 equation 8) 。證明在 tabular environment 下面，算法能收斂到 optimal multi-objetive policy
                - 原本的 single-value Q, 改為 sup over Qs in Pensieve
            - 論文 Figure 1: Q-pensieve 工作流程圖
            - 實作上很容易，且實證上發現不需要太大的 buffer size (size = 4)
            - 可能缺點: 疑似需要對每個 objetive dimension 都建立一個獨立的 Network                
                            
    
### 16 [劉育綸](https://www.cmlab.csie.ntu.edu.tw/~yulunliu/)
- 講題: Dynamic and Local Radiance Fields for Robust View Synthesis
- 時間安排關係沒有參加本 talk

## Day 2 (2023.03.25)

### 1 [李宏毅](https://speech.ee.ntu.edu.tw/~hylee/index.php)
- 講題: Foundation Model for Speech Processing
- 分享關於應用 Foundation Model 到 speech processing 上面的相關工作
    - 幾個基礎的工作
        - [2021: Self-Supervised Speech Representation Learning: A Review](https://arxiv.org/abs/2205.10643): 一篇 review
        - [2021: SUPERB: Speech processing Universal PERformance Benchmark](https://arxiv.org/abs/2105.01051): 語音 Benchmark 資料集
        - 宏毅老師的終極目標是開發出 End-to-End Speech Quetion Answering system，具有人類自然對話能力的機器人 "SpeechGPT"
    - 分享了兩篇近期的論文:
        - Paper 1: [Interspeech 2022: DUAL: Discrete Spoken Unit Adaptive Learning for Textless Spoken Question Answering](https://arxiv.org/abs/2203.04911)
            - Speech 領域也有對應的 Foundation Model, [2021: HuBERT](https://arxiv.org/abs/2106.07447), 但是他的 performance 並不像 NLP 領域，仍有極大的改善空間
            - A detour story: 在研究 DNA 序列分類問題的時候，發現把直接把 ATCG **隨機(!)** mapping 到 LLM 的一些 token 上面之後，再用 DNA data 去 fine-tune model 可以得到非常好的 performance 
                - 觀察: LLM 有極強的"序列結構"泛化能力 (這個 findings 很有趣也很驚人!)
            - 把上述技巧應用到 HuBERT 上面 (沒有去讀論文，但應該是把語音 token 隨機 map 到 LLM 的某些 token 上面) 可以得到極大的效能提升
        - Paper 2: Towards SpeechGPT 
            - 結束前稍微 promote 新的工作，沒有介紹細節
            - [paper](https://arxiv.org/abs/2203.16773), [github](https://github.com/ga642381/SpeechPrompt)

### 2 [陳縕儂](https://www.csie.ntu.edu.tw/~miulab/index.html)
- 講題: When Ads Meet Conversational Interfaces
- 演講內容
    - 介紹一篇研究: [ACL 2022: SalesBot: Transitioning from Chit-Chat to Task-Oriented Dialogues](https://arxiv.org/abs/2204.10591)
        - 工作目標: 訓練 chatbot 能夠從像業務人員一樣從 chitchat 了解客戶需求，在進而轉移到推銷模式，針對需求推薦產品
        - 兩個挑戰
            - When to transit ?(from chichat to task-oriented chat)
            - How to transit smoothly?
        - 解決方法:
            - 解決方案架構圖: 論文 Figure 2: 應用現存的 OPDG 和 TODG 模組再加上一個中間的切換機制
            - 切換機制: intent detector 在 chitchat 時期推論用戶需求
            - 利用 [OTTers](https://aclanthology.org/2021.acl-long.194/) datasets 去訓練 transition generation
        - 設計了三個 human rated scores (relevance, aggressiveness, overall) 評估對話生成結果品質

### 3 [楊奕軒](https://www.ee.ntu.edu.tw/bio1.php?id=1090726)
- 講題: Automatic Music Generation with the Transformers
- 演講內容:
    - 分享近年來 music generation 方法的演進
        - [2020: OpenAI Jukebox](https://openai.com/research/jukebox)
        - [2023: Google Music LLM](https://google-research.github.io/seanet/musiclm/examples/)
    - Taiwan AILab 的幾個音樂生成相關工作
        - AI Vocal 生成 demo & 陳珊妮[<教我如何做你的愛人>](https://www.cna.com.tw/news/ait/202303220359.aspx)歌聲生成
        - 沒有討論技術細節

### 4 [楊明玄](http://vllab.ucmerced.edu/)
- 講題: Learning to Synthesize Image and Video Contents
- 演講內容:
    - 介紹了講者多年以來關於 image synthesis 的相關工作，內容涉及至少 20-30 篇至 2016年左右以來的 paper 
    - 大致上有幾個應用方向
        - image-to-image translation
        - image in-painting / out-painting
            - [ILCR 2022: Infinity GAN](https://hubert0527.github.io/infinityGAN/)
        - image synthesis (generation)
            - [CVPR 2021: Lecam-GAN](https://github.com/google/lecam-gan): 改善 GAN 的 data efficiency
        - text-to-image synthesis 
        - text-to-image editting            
            - [2023: MUSE](https://muse-model.github.io/) (大作級)
                - 論文 figure 3, figure 4 模型架構流程圖
                - 三個模組: VQ-GAN tokenizer, Masked image model for sequntial image token prediction based on context and text info(transformer-type, most params from), low-to-high resolution (superres) translation transformer
                - 應用場景還很多，比如 text-to-image inpainting/outpainting
        - video synthesis
            - [CVPR 2023: MAGVIT](https://arxiv.org/abs/2212.05199) (大作級)

    - 大致上使用到的技巧關鍵字
        - VAE (latent space tricks), GAN, Diffusion        
        - VQ-VAE, VQ-GAN, VQ series for image tokenization        
        - Transformers (Masked Transformers), LLM
    
    - 內容太多待補完!


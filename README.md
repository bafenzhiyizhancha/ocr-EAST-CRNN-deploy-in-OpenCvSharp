# ocr-EAST-CRNN-deploy-in-OpenCvSharp
### EAST+CRNN模型的c#部署，使用OpenCvSharp进行模型加载与推理，OCR:EAST+CRNN模型用于字符的定位与识别
![捕获10](https://user-images.githubusercontent.com/26215301/157580491-77b3d494-e3db-425b-b3a7-5944ca964c6e.PNG)
![1](https://user-images.githubusercontent.com/26215301/157580507-08016fa6-b703-4530-8f80-82fb021b14b9.PNG)

注意事项：
1、src是在anycpu debug下编译的，如需切换到x64下编译，需要复制原debug下image于weights文件夹。
2、运行环境nuget包：OpenCvSharp与OpenCvSharp4.runtime.win

# COMP5421 Homework2 Report

----

## Problem 1.5 Key Point Detector

**The required image is attached below.**

**Solution: We utilize the np.roll function to create 10 separate matrices and then compare each of them. This could help us to eliminate the for loop.**

![](D:\Year3S\COMP5421\assignment\hw2_code_data\yyubm_report.assets\edge_suppresion.png)

Figure: With edge suppression



## Problem 2.4

See the figures below:



![chick_match2](D:\Year3S\COMP5421\assignment\hw2_code_data\yyubm_report.assets\chick_match1.png)

![chick_match4](D:\Year3S\COMP5421\assignment\hw2_code_data\yyubm_report.assets\chick_match2.png)



![chick_match3](D:\Year3S\COMP5421\assignment\hw2_code_data\yyubm_report.assets\chick_match3.png)

![](D:\Year3S\COMP5421\assignment\hw2_code_data\yyubm_report.assets\chick_match4.png)

![chick_match5](D:\Year3S\COMP5421\assignment\hw2_code_data\yyubm_report.assets\chick_match5.png)

**Inclined image**

![incline_match](D:\Year3S\COMP5421\assignment\hw2_code_data\yyubm_report.assets\incline_match.png)

**Books**



![desk](D:\Year3S\COMP5421\assignment\hw2_code_data\yyubm_report.assets\desk.png)

![floor](D:\Year3S\COMP5421\assignment\hw2_code_data\yyubm_report.assets\floor.png)

![floor_rot](D:\Year3S\COMP5421\assignment\hw2_code_data\yyubm_report.assets\floor_rot.png)

![plie](D:\Year3S\COMP5421\assignment\hw2_code_data\yyubm_report.assets\plie.png)

![stand](D:\Year3S\COMP5421\assignment\hw2_code_data\yyubm_report.assets\stand.png)

*Explanation:*

**We see that for chicken 5 and the books 3, 4 the match is not as good as before. This is mainly due to rotation of the features and many similar feature surroundings.** For the incline image, because they are parallel mostly, we see that the performance is quiet good. And the worst case is when we have similar features and rotation applied, the BRIEF is hard to tell them apart.



## Problem 2.5



![rotation_performance](D:\Year3S\COMP5421\assignment\hw2_code_data\yyubm_report.assets\rotation_performance.png)

>  We see that the number of matches drops significantly. And it increases at last. Because when degree is very small or very large, it is relatively similar to the origin image, but when the degree is in the middle, the image is largely rotated and the BRIEF could not recognize the  features well.



## Problem 3.

![文件_000](D:\Year3S\COMP5421\assignment\hw2_code_data\yyubm_report.assets\文件_000.jpeg)

 ## Problem 6.1

![q6_1](D:\Year3S\COMP5421\assignment\hw2_code_data\yyubm_report.assets\q6_1.jpg)

## Problem 6.2

![q6_2_pan](D:\Year3S\COMP5421\assignment\hw2_code_data\yyubm_report.assets\q6_2_pan.jpg)

## Problem 6.3

![q6_3](D:\Year3S\COMP5421\assignment\hw2_code_data\yyubm_report.assets\q6_3.jpg)

## Problem 7

![AR](D:\Year3S\COMP5421\assignment\hw2_code_data\yyubm_report.assets\AR-1616946893734.png)
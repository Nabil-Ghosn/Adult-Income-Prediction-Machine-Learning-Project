<h1>Adult-Income-Prediction-Machine-Learning-Project</h1>
Achieving the most accurate predictive model for determining whether an individual’s income exceeds $50,000 per annum, machine learning algorithms are employed on the individuals’ income database. The primary objective is to develop a robust model that can effectively predict the income of a given person based on the provided data.


<h2>Dataset</h2>

<p>We will work on the adult dataset provided by the
UCI1 website, which was collected by the United States Census Bureau (USCB),
responsible for collecting demographic and economic information about
individuals.</p>

<p><a href="http://archive.ics.uci.edu/dataset/2/adult1">http://archive.ics.uci.edu/dataset/2/adult</a></p>

<p>The database contains about 49,000 samples and
includes 14 characteristics to describe individuals, which we will mention in
the following table:</p>

<div>

<table style='border-collapse:collapse;border:none;margin-left:6.75pt;
 margin-right:6.75pt'>
 <tr style='height:28.1pt'>
  <td style='width:91.8pt;border:solid #BFBFBF 1.0pt;
  padding:0in 5.4pt 0in 5.4pt;height:28.1pt'>
  <p style='margin-bottom:0in;line-height:normal'><b>Type</b></p>
  </td>
  <td style='width:160.7pt;border:solid #BFBFBF 1.0pt;
  border-right:none;padding:0in 5.4pt 0in 5.4pt;height:28.1pt'>
  <p style='margin-bottom:0in;line-height:normal'><b>Feature</b></p>
  </td>
  <td style='width:.95in;border:solid #BFBFBF 1.0pt;
  border-right:none;padding:0in 5.4pt 0in 5.4pt;height:28.1pt'>
  <p style='margin-bottom:0in;line-height:normal'><b>Id</b></p>
  </td>
 </tr>
 <tr style='height:26.75pt'>
  <td style='width:91.8pt;border:solid #BFBFBF 1.0pt;
  border-top:none;background:#F2F2F2;padding:0in 5.4pt 0in 5.4pt;height:26.75pt'>
  <p style='margin-bottom:0in;line-height:normal'><b><span
  style='color:black'>numeric</span></b></p>
  </td>
  <td style='width:160.7pt;border-top:none;border-left:
  solid #BFBFBF 1.0pt;border-bottom:solid #BFBFBF 1.0pt;border-right:none;
  background:#F2F2F2;padding:0in 5.4pt 0in 5.4pt;height:26.75pt'>
  <p style='margin-bottom:0in;line-height:normal'><span
  style='color:black'>Age</span></p>
  </td>
  <td style='width:.95in;border-top:none;border-left:solid #BFBFBF 1.0pt;
  border-bottom:solid #BFBFBF 1.0pt;border-right:none;background:#F2F2F2;
  padding:0in 5.4pt 0in 5.4pt;height:26.75pt'>
  <p style='margin-bottom:0in;line-height:normal'><span
  style='color:black'>1</span></p>
  </td>
 </tr>
 <tr style='height:26.05pt'>
  <td style='width:91.8pt;border:solid #BFBFBF 1.0pt;
  border-top:none;padding:0in 5.4pt 0in 5.4pt;height:26.05pt'>
  <p style='margin-bottom:0in;line-height:normal'><b>categorical</b></p>
  </td>
  <td style='width:160.7pt;border-top:none;border-left:
  solid #BFBFBF 1.0pt;border-bottom:solid #BFBFBF 1.0pt;border-right:none;
  padding:0in 5.4pt 0in 5.4pt;height:26.05pt'>
  <p style='margin-bottom:0in;line-height:normal'>workclass</p>
  </td>
  <td style='width:.95in;border-top:none;border-left:solid #BFBFBF 1.0pt;
  border-bottom:solid #BFBFBF 1.0pt;border-right:none;padding:0in 5.4pt 0in 5.4pt;
  height:26.05pt'>
  <p style='margin-bottom:0in;line-height:normal'>2</p>
  </td>
 </tr>
 <tr style='height:26.75pt'>
  <td style='width:91.8pt;border:solid #BFBFBF 1.0pt;
  border-top:none;background:#F2F2F2;padding:0in 5.4pt 0in 5.4pt;height:26.75pt'>
  <p style='margin-bottom:0in;line-height:normal'><b><span
  style='color:black'>numeric</span></b></p>
  </td>
  <td style='width:160.7pt;border-top:none;border-left:
  solid #BFBFBF 1.0pt;border-bottom:solid #BFBFBF 1.0pt;border-right:none;
  background:#F2F2F2;padding:0in 5.4pt 0in 5.4pt;height:26.75pt'>
  <p style='margin-bottom:0in;line-height:normal'><span
  style='color:black'>fnlwgt</span></p>
  </td>
  <td style='width:.95in;border-top:none;border-left:solid #BFBFBF 1.0pt;
  border-bottom:solid #BFBFBF 1.0pt;border-right:none;background:#F2F2F2;
  padding:0in 5.4pt 0in 5.4pt;height:26.75pt'>
  <p style='margin-bottom:0in;line-height:normal'><span
  lang=AR-SY style='color:black'>3</span></p>
  </td>
 </tr>
 <tr style='height:26.05pt'>
  <td style='width:91.8pt;border:solid #BFBFBF 1.0pt;
  border-top:none;padding:0in 5.4pt 0in 5.4pt;height:26.05pt'>
  <p style='margin-bottom:0in;line-height:normal'><b>categorical</b></p>
  </td>
  <td style='width:160.7pt;border-top:none;border-left:
  solid #BFBFBF 1.0pt;border-bottom:solid #BFBFBF 1.0pt;border-right:none;
  padding:0in 5.4pt 0in 5.4pt;height:26.05pt'>
  <p style='margin-bottom:0in;line-height:normal'>education</p>
  </td>
  <td style='width:.95in;border-top:none;border-left:solid #BFBFBF 1.0pt;
  border-bottom:solid #BFBFBF 1.0pt;border-right:none;padding:0in 5.4pt 0in 5.4pt;
  height:26.05pt'>
  <p style='margin-bottom:0in;line-height:normal'><span
  lang=AR-SY>4</span></p>
  </td>
 </tr>
 <tr style='height:26.75pt'>
  <td style='width:91.8pt;border:solid #BFBFBF 1.0pt;
  border-top:none;background:#F2F2F2;padding:0in 5.4pt 0in 5.4pt;height:26.75pt'>
  <p style='margin-bottom:0in;line-height:normal'><b><span
  style='color:black'>numeric</span></b></p>
  </td>
  <td style='width:160.7pt;border-top:none;border-left:
  solid #BFBFBF 1.0pt;border-bottom:solid #BFBFBF 1.0pt;border-right:none;
  background:#F2F2F2;padding:0in 5.4pt 0in 5.4pt;height:26.75pt'>
  <p style='margin-bottom:0in;line-height:normal'><span
  style='color:black'>education_num</span></p>
  </td>
  <td style='width:.95in;border-top:none;border-left:solid #BFBFBF 1.0pt;
  border-bottom:solid #BFBFBF 1.0pt;border-right:none;background:#F2F2F2;
  padding:0in 5.4pt 0in 5.4pt;height:26.75pt'>
  <p style='margin-bottom:0in;line-height:normal'><span
  lang=AR-SY style='color:black'>5</span></p>
  </td>
 </tr>
 <tr style='height:26.75pt'>
  <td style='width:91.8pt;border:solid #BFBFBF 1.0pt;
  border-top:none;padding:0in 5.4pt 0in 5.4pt;height:26.75pt'>
  <p style='margin-bottom:0in;line-height:normal'><b>categorical</b></p>
  </td>
  <td style='width:160.7pt;border-top:none;border-left:
  solid #BFBFBF 1.0pt;border-bottom:solid #BFBFBF 1.0pt;border-right:none;
  padding:0in 5.4pt 0in 5.4pt;height:26.75pt'>
  <p style='margin-bottom:0in;line-height:normal'>martial_status</p>
  </td>
  <td style='width:.95in;border-top:none;border-left:solid #BFBFBF 1.0pt;
  border-bottom:solid #BFBFBF 1.0pt;border-right:none;padding:0in 5.4pt 0in 5.4pt;
  height:26.75pt'>
  <p style='margin-bottom:0in;line-height:normal'><span
  lang=AR-SY>6</span></p>
  </td>
 </tr>
 <tr style='height:26.05pt'>
  <td style='width:91.8pt;border:solid #BFBFBF 1.0pt;
  border-top:none;background:#F2F2F2;padding:0in 5.4pt 0in 5.4pt;height:26.05pt'>
  <p style='margin-bottom:0in;line-height:normal'><b><span
  style='color:black'>categorical</span></b></p>
  </td>
  <td style='width:160.7pt;border-top:none;border-left:
  solid #BFBFBF 1.0pt;border-bottom:solid #BFBFBF 1.0pt;border-right:none;
  background:#F2F2F2;padding:0in 5.4pt 0in 5.4pt;height:26.05pt'>
  <p style='margin-bottom:0in;line-height:normal'><span
  style='color:black'>occupation</span></p>
  </td>
  <td style='width:.95in;border-top:none;border-left:solid #BFBFBF 1.0pt;
  border-bottom:solid #BFBFBF 1.0pt;border-right:none;background:#F2F2F2;
  padding:0in 5.4pt 0in 5.4pt;height:26.05pt'>
  <p style='margin-bottom:0in;line-height:normal'><span
  lang=AR-SY style='color:black'>7</span></p>
  </td>
 </tr>
 <tr style='height:26.75pt'>
  <td style='width:91.8pt;border:solid #BFBFBF 1.0pt;
  border-top:none;padding:0in 5.4pt 0in 5.4pt;height:26.75pt'>
  <p style='margin-bottom:0in;line-height:normal'><b>categorical</b></p>
  </td>
  <td style='width:160.7pt;border-top:none;border-left:
  solid #BFBFBF 1.0pt;border-bottom:solid #BFBFBF 1.0pt;border-right:none;
  padding:0in 5.4pt 0in 5.4pt;height:26.75pt'>
  <p style='margin-bottom:0in;line-height:normal'>relationship</p>
  </td>
  <td style='width:.95in;border-top:none;border-left:solid #BFBFBF 1.0pt;
  border-bottom:solid #BFBFBF 1.0pt;border-right:none;padding:0in 5.4pt 0in 5.4pt;
  height:26.75pt'>
  <p style='margin-bottom:0in;line-height:normal'><span></span><span lang=AR-SY><span></span>8</span></p>
  </td>
 </tr>
 <tr style='height:26.75pt'>
  <td style='width:91.8pt;border:solid #BFBFBF 1.0pt;
  border-top:none;background:#F2F2F2;padding:0in 5.4pt 0in 5.4pt;height:26.75pt'>
  <p style='margin-bottom:0in;line-height:normal'><b><span
  style='color:black'>categorical</span></b></p>
  </td>
  <td style='width:160.7pt;border-top:none;border-left:
  solid #BFBFBF 1.0pt;border-bottom:solid #BFBFBF 1.0pt;border-right:none;
  background:#F2F2F2;padding:0in 5.4pt 0in 5.4pt;height:26.75pt'>
  <p style='margin-bottom:0in;line-height:normal'><span
  style='color:black'>race</span></p>
  </td>
  <td style='width:.95in;border-top:none;border-left:solid #BFBFBF 1.0pt;
  border-bottom:solid #BFBFBF 1.0pt;border-right:none;background:#F2F2F2;
  padding:0in 5.4pt 0in 5.4pt;height:26.75pt'>
  <p style='margin-bottom:0in;line-height:normal'><span
></span><span lang=AR-SY style='color:black'><span></span>9</span></p>
  </td>
 </tr>
 <tr style='height:26.05pt'>
  <td style='width:91.8pt;border:solid #BFBFBF 1.0pt;
  border-top:none;padding:0in 5.4pt 0in 5.4pt;height:26.05pt'>
  <p style='margin-bottom:0in;line-height:normal'><b>categorical</b></p>
  </td>
  <td style='width:160.7pt;border-top:none;border-left:
  solid #BFBFBF 1.0pt;border-bottom:solid #BFBFBF 1.0pt;border-right:none;
  padding:0in 5.4pt 0in 5.4pt;height:26.05pt'>
  <p style='margin-bottom:0in;line-height:normal'>sex</p>
  </td>
  <td style='width:.95in;border-top:none;border-left:solid #BFBFBF 1.0pt;
  border-bottom:solid #BFBFBF 1.0pt;border-right:none;padding:0in 5.4pt 0in 5.4pt;
  height:26.05pt'>
  <p style='margin-bottom:0in;line-height:normal'><span
></span><span lang=AR-SY><span></span>10</span></p>
  </td>
 </tr>
 <tr style='height:26.75pt'>
  <td style='width:91.8pt;border:solid #BFBFBF 1.0pt;
  border-top:none;background:#F2F2F2;padding:0in 5.4pt 0in 5.4pt;height:26.75pt'>
  <p style='margin-bottom:0in;line-height:normal'><b><span
  style='color:black'>numeric</span></b></p>
  </td>
  <td style='width:160.7pt;border-top:none;border-left:
  solid #BFBFBF 1.0pt;border-bottom:solid #BFBFBF 1.0pt;border-right:none;
  background:#F2F2F2;padding:0in 5.4pt 0in 5.4pt;height:26.75pt'>
  <p style='margin-bottom:0in;line-height:normal'><span
  style='color:black'>capital_gain</span></p>
  </td>
  <td style='width:.95in;border-top:none;border-left:solid #BFBFBF 1.0pt;
  border-bottom:solid #BFBFBF 1.0pt;border-right:none;background:#F2F2F2;
  padding:0in 5.4pt 0in 5.4pt;height:26.75pt'>
  <p style='margin-bottom:0in;line-height:normal'><span
></span><span lang=AR-SY style='color:black'><span></span>11</span></p>
  </td>
 </tr>
 <tr style='height:26.05pt'>
  <td style='width:91.8pt;border:solid #BFBFBF 1.0pt;
  border-top:none;padding:0in 5.4pt 0in 5.4pt;height:26.05pt'>
  <p style='margin-bottom:0in;line-height:normal'><b>numeric</b></p>
  </td>
  <td style='width:160.7pt;border-top:none;border-left:
  solid #BFBFBF 1.0pt;border-bottom:solid #BFBFBF 1.0pt;border-right:none;
  padding:0in 5.4pt 0in 5.4pt;height:26.05pt'>
  <p style='margin-bottom:0in;line-height:normal'>capital_loss</p>
  </td>
  <td style='width:.95in;border-top:none;border-left:solid #BFBFBF 1.0pt;
  border-bottom:solid #BFBFBF 1.0pt;border-right:none;padding:0in 5.4pt 0in 5.4pt;
  height:26.05pt'>
  <p style='margin-bottom:0in;line-height:normal'><span
></span><span lang=AR-SY><span></span>12</span></p>
  </td>
 </tr>
 <tr style='height:26.75pt'>
  <td style='width:91.8pt;border:solid #BFBFBF 1.0pt;
  border-top:none;background:#F2F2F2;padding:0in 5.4pt 0in 5.4pt;height:26.75pt'>
  <p style='margin-bottom:0in;line-height:normal'><b><span
  style='color:black'>numeric</span></b></p>
  </td>
  <td style='width:160.7pt;border-top:none;border-left:
  solid #BFBFBF 1.0pt;border-bottom:solid #BFBFBF 1.0pt;border-right:none;
  background:#F2F2F2;padding:0in 5.4pt 0in 5.4pt;height:26.75pt'>
  <p style='margin-bottom:0in;line-height:normal'><span
  style='color:black'>hours_per_week</span></p>
  </td>
  <td style='width:.95in;border-top:none;border-left:solid #BFBFBF 1.0pt;
  border-bottom:solid #BFBFBF 1.0pt;border-right:none;background:#F2F2F2;
  padding:0in 5.4pt 0in 5.4pt;height:26.75pt'>
  <p style='margin-bottom:0in;line-height:normal'><span
></span><span lang=AR-SY style='color:black'><span></span>13</span></p>
  </td>
 </tr>
 <tr style='height:26.05pt'>
  <td style='width:91.8pt;border:solid #BFBFBF 1.0pt;
  border-top:none;padding:0in 5.4pt 0in 5.4pt;height:26.05pt'>
  <p style='margin-bottom:0in;line-height:normal'><b>categorical</b></p>
  </td>
  <td style='width:160.7pt;border-top:none;border-left:
  solid #BFBFBF 1.0pt;border-bottom:solid #BFBFBF 1.0pt;border-right:none;
  padding:0in 5.4pt 0in 5.4pt;height:26.05pt'>
  <p style='margin-bottom:0in;line-height:normal'>native_country</p>
  </td>
  <td style='width:.95in;border-top:none;border-left:solid #BFBFBF 1.0pt;
  border-bottom:solid #BFBFBF 1.0pt;border-right:none;padding:0in 5.4pt 0in 5.4pt;
  height:26.05pt'>
  <p style='margin-bottom:0in;line-height:normal'><span
></span><span lang=AR-SY><span></span>14</span></p>
  </td>
 </tr>
</table>

</div>
</div>

<div>

<h2>Preprocessing</h2>

<p>First, Removing duplicate values from the database,
about 24 values. Then calculating the missing values as a percentage and replacing
them with the most frequent value (mode) of the corresponding attribute.</p>

<p>The features are mainly divided into six numerical
and eight categorical features. </p>

<p>According to the numerical features, trying to
explore and study the linear relationship and showing a heat map of the linear
correlation matrix to help identify the relation of every pair of features in
order to eliminate one of them.</p>

<p>Performing a standardization process (z-score
normalization) for numerical features due to their importance for algorithms
such as logistic regression.</p>

<p>Transforming numerical features to categorical by
digitizing, but it turned out to be unhelpful because the data is not normally
distributed and its order is important.</p>

<p>For the categorical features, representing them as
vectors with one-hot encoding, removing the low frequencies for the categories
(which contain zeros at a high rate of more than 99 percent), then deleting the
first vector of each feature, which can be represented in other categories as
zeros.</p>

<p style='text-align:center;line-height:
150%;page-break-after:avoid'><span lang=AR-SY><img src="Income%20prediction_files/image001.png"></span></p>

<p style='text-align:center'><span
style='font-size:14.0pt'>figure </span><span
style='font-size:14.0pt'>1</span></p>

<h2>Data Exploration</h2>

<p>Based on the heat map in Figure 1, we studied the
relationship between pairs of features in order to retain the important
features in the training process. We noticed that the relationship between the
pairs is generally weak, so we cannot get rid of one of the pairs and be
satisfied with the other. Also from Figure 1, we notice that there is a weak
correlation between the target and the fnlwgt attribute, which will be
suggested to remove it after ensuring that it will not affect the accuracy of
the training process.</p>

<p>For categorical features, we removed several features,
such as the education feature, which has an equivalent attribute,
education_num, which is numerical, and other features such as native_country,
race, and sex, based on the experimentation process.</p>

<h2>Testing</h2>

<p>Splitting the dataset into 80% for fine tuning, and
20% for final testing.</p>

<p>For the baseline model, which is a dummy classifier
model based on the mode, to determine the minimum expected performance, we
obtained an accuracy of 75%. We can explain the result easily by the imbalance
of the data.</p>

<p>Because of the imbalance of the dataset, we need another
criterion beside accuracy to evaluate each model, so we adopted the area under
curve (ROC) in the optimization process to reach the best hyperparameters of
models. </p>

<p>In the following table, the accuracy of each model
is shown according to the training dataset. </p>

<div>

<table
 style='border-collapse:collapse;border:none'>
 <tr>
  <td style='width:137.6pt;border:solid windowtext 1.0pt;
  padding:0in 5.4pt 0in 5.4pt'>
  <p style='margin-bottom:0in;text-align:
  center;line-height:normal'><b>ROC</b></p>
  </td>
  <td style='width:150.6pt;border:solid windowtext 1.0pt;
  border-right:none;padding:0in 5.4pt 0in 5.4pt'>
  <p style='margin-bottom:0in;text-align:
  center;line-height:normal'><b>Accuracy</b></p>
  </td>
  <td style='width:148.85pt;border:solid windowtext 1.0pt;
  border-right:none;padding:0in 5.4pt 0in 5.4pt'>
  <p style='margin-bottom:0in;text-align:
  center;line-height:normal'><b>Model</b></p>
  </td>
 </tr>
 <tr>
  <td style='width:137.6pt;border:solid windowtext 1.0pt;
  border-top:none;padding:0in 5.4pt 0in 5.4pt'>
  <p style='margin-bottom:0in;text-align:
  center;line-height:normal'>-</p>
  </td>
  <td style='width:150.6pt;border-top:none;border-left:
  solid windowtext 1.0pt;border-bottom:solid windowtext 1.0pt;border-right:
  none;padding:0in 5.4pt 0in 5.4pt'>
  <p style='margin-bottom:0in;text-align:
  center;line-height:normal'>75%</p>
  </td>
  <td style='width:148.85pt;border-top:none;border-left:
  solid windowtext 1.0pt;border-bottom:solid windowtext 1.0pt;border-right:
  none;padding:0in 5.4pt 0in 5.4pt'>
  <p style='margin-bottom:0in;text-align:
  center;line-height:normal'>Dummy classifier</p>
  </td>
 </tr>
 <tr>
  <td style='width:137.6pt;border:solid windowtext 1.0pt;
  border-top:none;padding:0in 5.4pt 0in 5.4pt'>
  <p style='margin-bottom:0in;text-align:
  center;line-height:normal'>0.911</p>
  </td>
  <td style='width:150.6pt;border-top:none;border-left:
  solid windowtext 1.0pt;border-bottom:solid windowtext 1.0pt;border-right:
  none;padding:0in 5.4pt 0in 5.4pt'>
  <p style='margin-bottom:0in;text-align:
  center;line-height:normal'>85%</p>
  </td>
  <td style='width:148.85pt;border-top:none;border-left:
  solid windowtext 1.0pt;border-bottom:solid windowtext 1.0pt;border-right:
  none;padding:0in 5.4pt 0in 5.4pt'>
  <p style='margin-bottom:0in;text-align:
  center;line-height:normal'>Logistic regression</p>
  </td>
 </tr>
 <tr>
  <td style='width:137.6pt;border:solid windowtext 1.0pt;
  border-top:none;padding:0in 5.4pt 0in 5.4pt'>
  <p style='margin-bottom:0in;text-align:
  center;line-height:normal'>0.905</p>
  </td>
  <td style='width:150.6pt;border-top:none;border-left:
  solid windowtext 1.0pt;border-bottom:solid windowtext 1.0pt;border-right:
  none;padding:0in 5.4pt 0in 5.4pt'>
  <p style='margin-bottom:0in;text-align:
  center;line-height:normal'>85%</p>
  </td>
  <td style='width:148.85pt;border-top:none;border-left:
  solid windowtext 1.0pt;border-bottom:solid windowtext 1.0pt;border-right:
  none;padding:0in 5.4pt 0in 5.4pt'>
  <p style='margin-bottom:0in;text-align:
  center;line-height:normal'>K-nearest neighbor</p>
  </td>
 </tr>
 <tr>
  <td style='width:137.6pt;border:solid windowtext 1.0pt;
  border-top:none;padding:0in 5.4pt 0in 5.4pt'>
  <p style='margin-bottom:0in;text-align:
  center;line-height:normal'>0.896</p>
  </td>
  <td style='width:150.6pt;border-top:none;border-left:
  solid windowtext 1.0pt;border-bottom:solid windowtext 1.0pt;border-right:
  none;padding:0in 5.4pt 0in 5.4pt'>
  <p style='margin-bottom:0in;text-align:
  center;line-height:normal'>85%</p>
  </td>
  <td style='width:148.85pt;border-top:none;border-left:
  solid windowtext 1.0pt;border-bottom:solid windowtext 1.0pt;border-right:
  none;padding:0in 5.4pt 0in 5.4pt'>
  <p style='margin-bottom:0in;text-align:
  center;line-height:normal'>Decision tree</p>
  </td>
 </tr>
 <tr>
  <td style='width:137.6pt;border:solid windowtext 1.0pt;
  border-top:none;padding:0in 5.4pt 0in 5.4pt'>
  <p style='margin-bottom:0in;text-align:
  center;line-height:normal'>0.926</p>
  </td>
  <td style='width:150.6pt;border-top:none;border-left:
  solid windowtext 1.0pt;border-bottom:solid windowtext 1.0pt;border-right:
  none;padding:0in 5.4pt 0in 5.4pt'>
  <p style='margin-bottom:0in;text-align:
  center;line-height:normal'>86.7%</p>
  </td>
  <td style='width:148.85pt;border-top:none;border-left:
  solid windowtext 1.0pt;border-bottom:solid windowtext 1.0pt;border-right:
  none;padding:0in 5.4pt 0in 5.4pt'>
  <p style='margin-bottom:0in;text-align:
  center;line-height:normal'>AdaBoost</p>
  </td>
 </tr>
 <tr>
  <td style='width:137.6pt;border:solid windowtext 1.0pt;
  border-top:none;padding:0in 5.4pt 0in 5.4pt'>
  <p style='margin-bottom:0in;text-align:
  center;line-height:normal'>0.929</p>
  </td>
  <td style='width:150.6pt;border-top:none;border-left:
  solid windowtext 1.0pt;border-bottom:solid windowtext 1.0pt;border-right:
  none;padding:0in 5.4pt 0in 5.4pt'>
  <p style='margin-bottom:0in;text-align:
  center;line-height:normal'>86.7%</p>
  </td>
  <td style='width:148.85pt;border-top:none;border-left:
  solid windowtext 1.0pt;border-bottom:solid windowtext 1.0pt;border-right:
  none;padding:0in 5.4pt 0in 5.4pt'>
  <p style='margin-bottom:0in;text-align:
  center;line-height:normal'>Gradient Boosting</p>
  </td>
 </tr>
</table>

</div>

<p>The best model that achieved the highest accuracy
and highest degree of generalization, which is GBoost, with a maximum depth
equals to 4 and a number of estimators equal to 250, the table below shows the
results were obtained for the best model based on testing dataset:</p>

<div>

<table
 style='border-collapse:collapse;border:none'>
 <tr>
  <td style='width:112.7pt;border:solid windowtext 1.0pt;
  padding:0in 5.4pt 0in 5.4pt'>
  <p style='margin-bottom:0in;text-align:
  center;line-height:normal'><b>AUC</b></p>
  </td>
  <td style='width:112.7pt;border:solid windowtext 1.0pt;
  border-right:none;padding:0in 5.4pt 0in 5.4pt'>
  <p style='margin-bottom:0in;text-align:
  center;line-height:normal'><b>Precision</b></p>
  </td>
  <td style='width:112.7pt;border:solid windowtext 1.0pt;
  border-right:none;padding:0in 5.4pt 0in 5.4pt'>
  <p style='margin-bottom:0in;text-align:
  center;line-height:normal'><b>Recall</b></p>
  </td>
  <td style='width:112.7pt;border:solid windowtext 1.0pt;
  border-right:none;padding:0in 5.4pt 0in 5.4pt'>
  <p style='margin-bottom:0in;text-align:
  center;line-height:normal'><b>Accuracy</b></p>
  </td>
 </tr>
 <tr>
  <td style='width:112.7pt;border:solid windowtext 1.0pt;
  border-top:none;padding:0in 5.4pt 0in 5.4pt'>
  <p style='margin-bottom:0in;text-align:
  center;line-height:normal'><span></span><span lang=AR-SY
  style='font-size:10.0pt;font-family:"Arial",sans-serif'><span></span>0.9261942966661167</span></p>
  </td>
  <td style='width:112.7pt;border-top:none;border-left:
  solid windowtext 1.0pt;border-bottom:solid windowtext 1.0pt;border-right:
  none;padding:0in 5.4pt 0in 5.4pt'>
  <p style='margin-bottom:0in;text-align:
  center;line-height:normal'><span lang=AR-SY style='font-size:10.0pt;
  font-family:"Arial",sans-serif'>0.779771110423755</span></p>
  </td>
  <td style='width:112.7pt;border-top:none;border-left:
  solid windowtext 1.0pt;border-bottom:solid windowtext 1.0pt;border-right:
  none;padding:0in 5.4pt 0in 5.4pt'>
  <p style='margin-bottom:0in;text-align:
  center;line-height:normal'><span lang=AR-SY style='font-size:10.0pt;
  font-family:"Arial",sans-serif'>0.6554862194487779</span></p>
  </td>
  <td style='width:112.7pt;border-top:none;border-left:
  solid windowtext 1.0pt;border-bottom:solid windowtext 1.0pt;border-right:
  none;padding:0in 5.4pt 0in 5.4pt'>
  <p style='margin-bottom:0in;text-align:
  center;line-height:normal'><span></span><span style='font-size:10.0pt;
  font-family:"Arial",sans-serif'><span></span>0.8748848350838401</span></p>
  </td>
 </tr>
</table>

</div>

<p>Here are two figures, the Rock curve and the
confusion matrix. The confusion matrix represents horizontally the actual
distribution, and vertically what was classified by the model, and the darkness
of the secondary diameter indicates that the model made good prediction on the
test data.</p>

<p style='text-align:center;line-height:
150%;page-break-after:avoid'><span style='font-size:16.0pt;line-height:150%'><img id="Picture 2"
src="Income%20prediction_files/image002.png"></span></p>

<p style='text-align:center'>figure 2</p>

<p style='text-align:center;line-height:
150%;page-break-after:avoid'><span style='font-size:16.0pt;line-height:150%'><img
id="Picture 1"
src="Income%20prediction_files/image003.png"></span></p>

<p style='text-align:center'>figure 3</p>

<h2>Conclusion</h2>

<p>In this research, we tested a number of automatic
learning models and documented the accuracy of each algorithm in the table
above. In conclusion, we would like to clarify that the dataset is specific to
individuals in the United States of America. So the results are not generally
accurate due to the presence of several standards and factors related to the
income of individuals depending on the geography.</p>

</div>

<object data="/Adult%20Income%20Prediction.pdf" type="application/pdf" width="700px" height="700px">
  <embed src="/Adult%20Income%20Prediction.pdf">
  <p>You can download the presentation: <a href="/Adult%20Income%20Prediction.pdf">Download PDF</a>.</p>
  </embed>
</object>

<h1>
    EigenGame 
</h1>
<h3>
    Contributors
</h3>
<ul>
    <li>
        <strong>Rishanth .R</strong>
    </li>
</ul>
<p>
    This is an implementation of the EigenGame proposed in the paper <a href="https://arxiv.org/abs/2010.00554">"EigenGame: PCA as a Nash Equilibrium" by Gemp et. al.</a>.
</p>
<p>
    <h4>
        Running the program
    </h4>
    Run the program with the flag "-generateX" if you want to generate X before playing the eigengame<br/>
    The "-printX" flag should be used to print the X being used in the eigenGame<br/>
    The "-repeatedEVtest" flag should be used to use the hard-coded matrix X with repeated eigenvalues for the eigenGame<br/>
    The major hyperparameters of the game (dimensions of X, number of top eigenvectors to find, number of iterations to run and the learning rate) are declared at the top of the program right after the module imports<br/>
    The program does not do any hyperparameter tuning<br/>
    It was observed that learning rate largely determined how close the solution obtained was to the actual eigenvectors<br/>
    Try experimenting with different powers of 10 for the learning rate<br/>
    The program has to be run with the "-modified" flag to use the modified update<br/>
    To continue the last played eigengame, use the "-continueEigenGame" flag<br/>
    The program stores players' positions (read calculated eigenvectors) after every iteration of update in the file "Vs.npy" (or "Vs_modified.npy" when used in conjunction with the "-modified" flag)<br/>
    The program stores total time elapsed since start of the game after every iteration of update in the file "iterTimes.npy" (or "iterTimes_modified.npy" when used in conjunction with the "-modified" flag)<br/>
    Distance between the calculated eigenvectors after every iteration of update during the eigengame and the actual eigenvectors (obtained through numpy) vs the number of iterations and total time elapsed since the start of the game can be plotted by using the "-analyseResults" flag<br/>
    When using the "-analyseResults" flag, if one wants to first play the eigengame, one has to also use the "-playEigenGame" flag while executing the program<br/> 
    The "-debug" flag can be used to print some debug information<br/>
    Arbitrary combination of flags might give undesirable results<br/>
    For example, using the "-repeatedEVtest" flag  along with the "analyseResults" flag without the "-playEigenGame" flag will only work as expected if the last played eigen game was with the "-repeatedEVtest" flag<br/>
    When run without flags, the program tries to get X from stored file in a hard-coded path. If not present, it would implicitly use the "-generateX" flag. The eigengame is then played and final results are printed<br/>
    All labels are case sensitive<br/>
</p>
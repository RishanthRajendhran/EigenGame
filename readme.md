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
    The major hyperparameters of the game (dimensions of X, number of top eigenvectors to find, number of iterations to run) are declared at the top of the program right after the module imports<br/>
    The program does not do any hyperparameter tuning<br/>
    It was observed that learning rate largely determined how close the solution obtained was to the actual eigenvectors<br/>
    Try experimenting with different powers of 10 for the learning rate<br/>
</p>
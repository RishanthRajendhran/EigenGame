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
        Files 
    </h4>
    <ul>
        <li>
            <strong>eigengame.py</strong><br/>
        </li>
        <li>
            <strong>eigengame_wandb.py</strong><br/>
            Every run of this code gets saved in wandb.
            Flags have to be manually added to the list "flags" in the hyperparameters_default dictionary. (Line Number: 12)
        </li>
        <li>
            <strong>sweep.yaml</strong><br/>
            Wandb sweep configuration file for "eigengame_wandb.py".
        </li>
    </ul>
</p>
<p>
    <h4>
        Variants of the EigenGame
    </h4>
    <ul>
        <li>
            <strong> Variant 1a </strong> - Asymmetric penalties and current player is allowed to finish the game before next player gets his/her/their chance
        </li>
        <li>
            <strong> Variant 2a </strong> - Symmetric penalties and current player is allowed to finish the game before next player gets his/her/their chance
        </li>
        <li>
            <strong> Variant 1b </strong> - Asymmetric penalties and current player is allowed to finish one step of the game before next player gets his/her/their chance
        </li>
        <li>
            <strong> Variant 2b </strong> - Symmetric penalties and current player is allowed to finish one step of the game before next player gets his/her/their chance
        </li>
        <li>
            <strong> Variant 1c </strong> - Asymmetric penalties and current player is allowed to finish L steps of the game before next player gets his/her/their chance where L is a hyperparameter such that 1 < L < numIterations
        </li>
        <li>
            <strong> Variant 2c </strong> - Symmetric penalties and current player is allowed to finish L steps of the game before next player gets his/her/their chance where  L is a hyperparameter such that 1 < L < numIterations
        </li>
    </ul>
</p>
<p>
    <h4>
        Variants of gradient ascent used in the EigenGame
    </h4>
    <p>
        In the paper, vanilla gradient ascent is used to perform the gradient updates. There are other variants of gradient ascent popular in machine learning literature such as momentum gradient ascenet, nesterov accelerated gradient ascent, Adaptive gradient ascent etc. Curretly, three such variants (momentum, rmsprop and adagrad) have been implemented in this program. 
    </p>
    <p>
        <h6>
            Momentum Gradient Ascent 
        </h6>
        <p>
            When performing gradient updates, we can keep track of the cummulative updates made until now and decide where to move based on this cummulative history and current gradient update. We calculate the momentum term (v_t) as follows: <br/>
            v_t = gamma*v_t-1 + learning_rate*gradTheta <br/>
            Theta can then be updates as: <br/>
            Theta = Theta - v_t <br/>
            Here gamma is a hyperparameter. 
        </p>
        <h6>
            RMSprop Gradient Ascent 
        </h6>
        <p>
            This variant attempts to adapt the learning rate to the parameters, performing smaller updates for frequently occuring features and larger updates for infrequent ones. This works well when working with sparse data. <br/>
            v_t = gamma*v_t-1 + (1-gamma)*(gradTheta**2) <br/>
            For feature i, the update is as follows: <br/>
            Theta_i = Theta_i - (learning_rate/sqrt(v_t_i + eps))*gradTheta_i <br/>
            Here gamma and eps are hyperparameters. 
        </p>
        <h6>
            Adagrad Gradient Ascent 
        </h6>
        <p>
            This variant attempts to adapt the learning rate to the parameters, performing smaller updates for frequently occuring features and larger updates for infrequent ones. This works well when working with sparse data. <br/>
            v_t = beta*v_t-1 + (1-beta)*(gradTheta) <br/>
            For feature i, the update is as follows: <br/>
            Theta_i = Theta_i - (learning_rate/sqrt(v_t_i + eps))*gradTheta_i <br/>
            Here beta and eps are hyperparameters. 
        </p>
    </p>
</p>
<p>
    <h4>
        Running the program
    </h4>
    The variant of the EigenGame to be played should be mentioned as a flag: "-variantA" (default) / "-variantB" / "-variantC" along with the "-symmetric" flag in case of the game with symmetric penalties<br/>
    When the "-symmetric" flag is not used, the EigenGame defaults to asymmetric penalties<br/>
    The variant of gradient ascent to be used should be specified as a flag: "-momentum"/"-rmsprop"/"-adagrad"<br/>
    By default, vanilla gradient ascent is performed<br/>
    Run the program with the flag "-generateX" if you want to generate X before playing the eigengame<br/>
    The "-printX" flag should be used to print the X being used in the eigenGame<br/>
    The "-repeatedEVtest" / "-repeatedEVtest2" flag should be used to use the hard-coded matrix X with repeated eigenvalues for the eigenGame<br/>
    The major hyperparameters of the game (dimensions of X, number of top eigenvectors to find, number of iterations, number of steps per iteration and the learning rate) are declared at the top of the program right after the module imports<br/>
    The program does not do any hyperparameter tuning<br/>
    It was observed that learning rate largely determined how close the solution obtained was to the actual eigenvectors<br/>
    Try experimenting with different powers of 10 for the learning rate<br/>
    The program has to be run with the "-symmetric" flag to use the symmetric penalties<br/>
    To continue the from where the last played eigengame stopped, use the "-continueEigenGame" flag<br/>
    The "-checkVectors" flag when used runs the checkVector routine after every update. The checkVector routine is intended to break up players moving in the same direction <br/>
    The program stores players' positions (read calculated eigenvectors) after every iteration of update in the file "Vs_{variant_here}.npy"<br/>
    The program stores total time elapsed since start of the game after every iteration of update in the file "iterTimes_{variant_here}.npy"<br/>
    Distance between the calculated eigenvectors after every iteration of update during the eigengame and the actual eigenvectors (obtained through numpy) vs the number of iterations and total time elapsed since the start of the game can be plotted by using the "-analyseResults" flag<br/>
    When using the "-analyseResults" flag, if one wants to first play the eigengame, one has to also use the "-playEigenGame" flag while executing the program<br/> 
    To save the plots generated by the "-analyseResults" flag, use the "-savePlots" flagMbr/>
    The "-debug" flag can be used to print some debug information<br/>
    The "-visualiseResults" flag when used in combination with the "-3D" flag can be used to visualise 3D eigenvectors after every iteration of the EigenGame. The "-saveVisualisations" flag can be used to save the visualisations<br/>
    The "-visualiseResultsTogether" flag when used in combination with the "-3D" flag can be used to visualise 3D eigenvectors after every iteration of the EigenGame all in the same plot.<br/>
    The deafault speed of these visualisations is "highSpeed". The flags "-mediumSpeed" and "-lowSpeed" can be used to slow down the visualisations appropriately<br/>
    The "-analyseAngles" flag can be used to generate plots of Angle between eigenvectors obtained through the eigengame and the expected eigenvectors (obtained through numpy) vs number of iterations/total time elapsed. The "-analyseAnglesTogether" flag will plot the angles for all eigenvectors in the same plot. The "-savePlots" can be used to save these plots.<br/>
    The "-saveMode" flag can be used while analysing results/angles and visualisations to only save them without showing them<br/>
    Arbitrary combination of flags might give undesirable results<br/>
    For example, using the "-repeatedEVtest" flag  along with the "analyseResults" flag without the "-playEigenGame" flag will only work as expected if the last played eigen game was with the "-repeatedEVtest" flag<br/>
    When run without flags, the program tries to get X from stored file in a hard-coded path. If not present, it would implicitly use the "-generateX" flag. The eigengame is then played and final results are printed<br/>
    All labels are case sensitive<br/>
</p>
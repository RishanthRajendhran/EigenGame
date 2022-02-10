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
            <strong> Variant 1c </strong> - Asymmetric penalties and current player is allowed to finish L steps of the game before next player gets his/her/their chance where L is a hyperparameter such that 1 < L < numIterations (It is assumed that L >= 25)
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
    Successful runs of the EigenGame (i.e. when players converge satisfactorily) are stored as a dictionary in the files "history.txt" and "history_{xDim}.txt" for later perusal<br/>
    Longest Correct EigenVectors Streak, defined as the longest streak of consecutive eigenvectors with angular errors less than a specified angularThreshold <em>(see arXiv:2010.00554 [cs.LG])</em>, can be computed by using the "-computeLCES" flag. LCES is computed for the last played EigenGame and LCES vs time elapsed is plotted. LCES at the last iteration is printed on the screen. <br/>
    Distance between the calculated eigenvectors after every iteration of update during the eigengame and the actual eigenvectors (obtained through numpy) vs the number of iterations and total time elapsed since the start of the game can be plotted by using the "-analyseResults" flag<br/>
    When using the "-analyseResults" flag, if one wants to first play the eigengame, one has to also use the "-playEigenGame" flag while executing the program<br/> 
    To save the plots generated by the "-analyseResults" flag, use the "-savePlots" flags<br/>
    The "-debug" flag can be used to print some debug information<br/>
    The "-visualiseResults" flag can be used to visualise the players in 3D after every iteration of the EigenGame. The "-saveVisualisations" flag can be used to save the visualisations<br/>
    The "-visualiseResultsTogether" flag can be used to visualise the players in 3D after every iteration of the EigenGame all in the same plot.<br/>
    The "-visualiseTrajectory" flag can be used to visualise the trajectory of the players in 3D after every iteration of the EigenGame. The "-saveVisualisations" flag can be used to save the visualisations<br/>
    The "-visualiseTrajectoryTogether" flag can be used to visualise the trajectory of the players in 3D after every iteration of the EigenGame all in the same plot.<br/>
    The deafault speed of these visualisations is "highSpeed". The flags "-mediumSpeed" and "-lowSpeed" can be used to slow down the visualisations appropriately.<br/>
    The default speed when "-saveVisualisations" or "-saveMode" flags are being used is "lowSpeed". If "highSpeed" is required, the flag "-highSpeed" should also be used.<br/>
    The "-analyseAngles" flag can be used to generate plots of Angle between eigenvectors obtained through the eigengame and the expected eigenvectors (obtained through numpy) vs number of iterations/total time elapsed. The "-analyseAnglesTogether" flag will plot the angles for all eigenvectors in the same plot. The "-savePlots" can be used to save these plots.<br/>
    The "-saveMode" flag can be used while analysing results/angles and visualisations to only save them without showing them<br/>
    The flag "-postGameAnalysis" can be used to produce the effect of the following flags at one go: "-analyseResults", "-analyseAngles", "-analyseAnglesTogether", "-visualiseResults", "-visualiseResultsTogether", "-visualiseTrajectory", "-visualiseTrajectoryTogether" and "-computeLCES".<br/>
    Arbitrary combination of flags might give undesirable results<br/>
    For example, using the "-repeatedEVtest" flag  along with the "analyseResults" flag without the "-playEigenGame" flag will only work as expected if the last played eigen game was with the "-repeatedEVtest" flag<br/>
    When run without flags, the program tries to get X from stored file in a hard-coded path. If not present, it would implicitly use the "-generateX" flag. The eigengame is then played and final results are printed<br/>
    All labels are case sensitive<br/>
    <h4>
        Summary of flags
    </h4>
    <ul>
        <li>
            <h6>
                -playEigenGame (default)
            </h6>
            <p>
                Plays the EigenGame 
            </p>
        </li>
        <li>
            <h6>
                -continueEigenGame
            </h6>
            <p>
                Continues the last played EigenGame
            </p>
        </li>
        <li>
            <h6>
                -repeatedEVtest / -repeatedEVtest2
            </h6>
            <p>
                Uses the hardcoded matrices with repeated EigenValues to play the EigenGame
            </p>
        </li>
        <li>
            <h6>
                -generateX
            </h6>
            <p>
                Generates a new matrix of specified dimensions and stores it in "X.npy" before playing the EigenGame
            </p>
        </li>
        <li>
            <h6>
                -printX
            </h6>
            <p>
                Prints the matrix in the "X.npy" file; this file contains the matrix being used in the current EigenGame, if the EigenGame is being played, or the matrix used in the last EigenGame, if the EigenGame is not being played. 
            </p>
        </li>
        <li>
            <h6>
                -debug
            </h6>
            <p>
                Prints degub information
            </p>
        </li>
        <li>
            <h6>
                -symmetric
            </h6>
            <p>
                Plays the symmetric EigenGame
            </p>
        </li>
        <li>
            <h6>
                -checkVectors
            </h6>
            <p>
                Performs check on the players after every iteration to see if they are closer than a specified threshold to other players and if they are, values (for that player) are reset to what it was in the previous iteration
            </p>
        </li>
        <li>
            <h6>
                -variantA (default) / -variantB / -variantC
            </h6>
            <p>
                Specifies the variant of the EigenGame 
            </p>
        </li>
        <li>
            <h6>
                -vanilla (default) / -momentum / -rmsprop / -adagrad
            </h6>
            <p>
                Specifies the gradient ascent variant
            </p>
        </li>
        <li>
            <h6>
                -computeLCES
            </h6>
            <p>
                Computes the Longest Common EigenVectors Streak for the last played EigenGame and shows the plot of LCES vs time elapsed
            </p>
        </li>
        <li>
            <h6>
                -analyseResults
            </h6>
            <p>
                Plots the (distance) vs (number of iterations / time elapsed) for the last played EigenGame
            </p>
        </li>
        <li>
            <h6>
                -analyseAngles
            </h6>
            <p>
                Plots the (angles) vs (time elapsed) for the last played EigenGame on per-player basis
            </p>
        </li>
        <li>
            <h6>
                -analyseAnglesTogether
            </h6>
            <p>
                Plots the (angles) vs (time elapsed) for the last played EigenGame for all players together in a single plot
            </p>
        </li>
        <li>
            <h6>
                -visualiseResults
            </h6>
            <p>
                Visualises in 3D the players across iterations of the last played EigenGame on per-player basis
            </p>
        </li>
        <li>
            <h6>
                -visualiseResultsTogether
            </h6>
            <p>
                Visualises in 3D the players across iterations of the last played EigenGame for all players together in a single visualisation
            </p>
        </li>
        <li>
            <h6>
                -visualiseTrajectory
            </h6>
            <p>
                Visualises in 3D the trajectory of the players across iterations of the last played EigenGame on per-player basis
            </p>
        </li>
        <li>
            <h6>
                -visualiseTrajectoryTogether
            </h6>
            <p>
                Visualises in 3D the trajectory of the players across iterations of the last played EigenGame for all players together in a single visualisation
            </p>
        </li>
        <li>
            <h6>
                -markPoints
            </h6>
            <p>
                Marks points in the trajectory visualisations using the 'x' marker 
            </p>
        </li>
        <li>
            <h6>
                -highSpeed (default) / -mediumSpeed / -lowSpeed
            </h6>
            <p>
                Specifies the speed of the visualisation
            </p>
        </li>
        <li>
            <h6>
                -analyseSubspaceAngles
            </h6>
            <p>
                Plots the (subspace angles) vs (time elapsed) for the last played EigenGame 
            </p>
        </li>
        <li>
            <h6>
                -postGameAnalysis
            </h6>
            <p>
                Shorthand flag to produce the effect of the following flags at one go:
                <ul>
                    <li>
                        "-analyseResults"
                    </li>
                    <li>
                        "-analyseAngles"
                    </li>
                    <li>
                        "-analyseAnglesTogether"
                    </li>
                    <li>
                        "-visualiseResults"
                    </li>
                    <li>
                        "-visualiseResultsTogether"
                    </li>
                    <li>
                        "-visualiseTrajeectory"
                    </li>
                    <li>
                        "-visualiseTrajectoryTogether"
                    </li>
                    <li>
                        "-computeLCES"
                    </li>
                </ul>
            </p>
        </li>
        <li>
            <h6>
                -savePlots
            </h6>
            <p>
                Saves the plots generated (if any) in the current run
            </p>
        </li>
        <li>
            <h6>
                -saveVisualisations
            </h6>
            <p>
                Saves the visualisations generated (if any) in the current run
            </p>
        </li>
        <li>
            <h6>
                -saveMode
            </h6>
            <p>
                Saves the plots and visualisations generated (if any) in the current run without showing them during the run
            </p>
        </li>
    </ul>
</p>
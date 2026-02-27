import '../styles/help.css';

/* ---------- short previews ---------- */

export function VotingIntroText({ effectiveTask, votingType }) {
	const isReg = effectiveTask === 'regression';

	return (
		<div className="helpStack helpStackTight">
			<div className="helpTitleStrong">Voting ensemble</div>

			<p className="helpTextBodyXs">
				Trains multiple base models on the same split and combines their predictions.
			</p>

			<p className="helpTextBodyXs">
				{isReg
					? 'Regression: averages predictions (VotingRegressor).'
					: votingType === 'soft'
					? 'Soft voting averages predicted probabilities.'
					: 'Hard voting chooses the majority class.'}
			</p>
		</div>
	);
}

export function BaggingIntroText({ effectiveTask }) {
	const isReg = effectiveTask === 'regression';

	return (
		<div className="helpStack helpStackTight">
			<div className="helpTitleStrong">Bagging ensemble</div>

			<p className="helpTextBodyXs">
				Trains many copies of the same estimator on resampled data and averages/votes.
			</p>

			<p className="helpTextBodyXs">
				{isReg
					? 'Regression: averages predictions (BaggingRegressor).'
					: 'Classification: majority vote across estimators (BaggingClassifier).'}
			</p>
		</div>
	);
}

export function AdaBoostIntroText({ effectiveTask }) {
	const isReg = effectiveTask === 'regression';

	return (
		<div className="helpStack helpStackTight">
			<div className="helpTitleStrong">AdaBoost ensemble</div>

			<p className="helpTextBodyXs">
				Adds weak learners sequentially, focusing more on the samples it previously got wrong.
			</p>

			<p className="helpTextBodyXs">
				{isReg
					? 'Regression: weighted ensemble of weak regressors (AdaBoostRegressor).'
					: 'Classification: boosts weak classifiers (AdaBoostClassifier).'}
			</p>
		</div>
	);
}

export function XGBoostIntroText({ effectiveTask }) {
	const isReg = effectiveTask === 'regression';

	return (
		<div className="helpStack helpStackTight">
			<div className="helpTitleStrong">XGBoost</div>

			<p className="helpTextBodyXs">
				Gradient boosted decision trees (high-performance for tabular data).
			</p>

			<p className="helpTextBodyXs">
				{isReg ? 'Regression: XGBRegressor.' : 'Classification: XGBClassifier.'}
			</p>
		</div>
	);
}

/* ---------- expanded help ---------- */

function VotingDetailsText({ effectiveTask, votingType }) {
	const isReg = effectiveTask === 'regression';

	return (
		<div className="helpStack helpStackXs">
			<div className="helpTitleStrong">How to choose settings</div>

			<ul className="helpList helpListXs helpListTight">
				<li>
					<span className="helpInlineLabel">Simple vs Advanced</span> – Simple uses default hyperparameters.
					Advanced lets you tune estimators and optionally set weights.
				</li>

				{!isReg && (
					<li>
						<span className="helpInlineLabel">Hard vs Soft</span> – Hard voting combines labels. Soft voting
						averages probabilities.
					</li>
				)}

				{!isReg && votingType === 'soft' && (
					<li>
						<span className="helpInlineLabel">Soft voting requirement</span> – all estimators must support{' '}
						<code className="helpInlineCode">predict_proba</code>.
					</li>
				)}

				<li>
					<span className="helpInlineLabel">Prefer diversity</span> – mixing model families often improves results.
				</li>

				<li>
					<span className="helpInlineLabel">Duplicates</span> – identical estimators act like implicit weighting;
					prefer explicit weights.
				</li>
			</ul>
		</div>
	);
}

function BaggingDetailsText() {
	return (
		<div className="helpStack helpStackXs">
			<div className="helpTitleStrong">How to choose settings</div>

			<ul className="helpList helpListXs helpListLoose">
				<li>
					<span className="helpInlineLabel">Number of estimators</span> – how many base models you train. More
					estimators usually reduce variance and make results more stable, but increase training time. Typical
					range: <span className="helpInlineStrong">25–200</span>. If results look noisy, increase this.
				</li>

				<li>
					<span className="helpInlineLabel">Max samples (fraction)</span> – fraction of the training fold used to
					train each estimator. <span className="helpInlineStrong">1.0</span> means “same size as the training
					fold” (with replacement if Bootstrap is on). Smaller values (e.g.{' '}
					<span className="helpInlineStrong">0.5–0.9</span>) increase diversity between estimators and can reduce
					overfitting, but each estimator learns from less data.
				</li>

				<li>
					<span className="helpInlineLabel">Max features (fraction)</span> – fraction of input features used per
					estimator (feature subsampling / random subspace). Lower values increase estimator diversity and can
					improve generalization in high-dimensional problems, but may reduce accuracy if too low. Common starting
					points: <span className="helpInlineStrong">0.5–1.0</span>.
				</li>

				<li>
					<span className="helpInlineLabel">Bootstrap</span> – when enabled, each estimator trains on a bootstrap sample
					(sampling <span className="helpInlineStrong">with replacement</span>). This is classic bagging and adds
					randomness. If disabled, sampling is <span className="helpInlineStrong">without replacement</span>. With
					Bootstrap off and Max samples = 1.0, every estimator sees the same rows (so only Max features adds
					randomness).
				</li>

				<li>
					<span className="helpInlineLabel">Bootstrap features</span> – when enabled, each estimator also trains on a
					resampled subset of features. This can further increase diversity, especially when you have many
					correlated features. If you already use Max features &lt; 1.0, this may be redundant.
				</li>

				<li>
					<span className="helpInlineLabel">Out-of-bag score</span> – only meaningful when{' '}
					<span className="helpInlineStrong">Bootstrap</span> is enabled. For each estimator, some training samples
					are not selected into its bootstrap sample (“out-of-bag” samples). The out-of-bag score evaluates
					predictions on those left-out samples and gives a built-in generalization estimate without creating a
					separate validation set. This is most useful for quick feedback and sanity checks; for reporting, prefer
					your chosen holdout / k-fold split.
				</li>

				<li>
					<span className="helpInlineLabel">Balanced bagging</span> – uses an imbalanced-learn variant that tries to
					reduce class imbalance inside each estimator’s training sample. Recommended when classes are noticeably
					imbalanced or when you see bagging failures due to class sparsity. If your dataset is only mildly
					imbalanced, you usually don’t need it.
				</li>

				<li>
					<span className="helpInlineLabel">Sampling strategy (Balanced bagging)</span> – controls how classes are
					balanced inside each bag. <span className="helpInlineStrong">Auto</span> is the safe default. Options like
					“majority”, “not minority”, etc. decide which classes are down-sampled. If you’re unsure, keep{' '}
					<span className="helpInlineStrong">Auto</span>.
				</li>

				<li>
					<span className="helpInlineLabel">Replacement (Balanced bagging)</span> – whether the class-balancing sampler
					is allowed to sample with replacement. Turning this on can help when some classes have few samples, but
					may increase duplicate rows inside a bag.
				</li>

				<li>
					<span className="helpInlineLabel">Number of jobs</span> – parallelism. Higher values use more CPU cores to
					train estimators faster. If supported, <span className="helpInlineStrong">-1</span> means “use all cores”.
					If your machine becomes unresponsive, reduce this.
				</li>

				<li>
					<span className="helpInlineLabel">Random state</span> – seed for reproducibility. Set it to a fixed value
					(e.g. 42) to make results repeatable across runs.
				</li>
			</ul>
		</div>
	);
}

function AdaBoostDetailsText({ effectiveTask }) {
	const isReg = effectiveTask === 'regression';

	return (
		<div className="helpStack helpStackXs">
			<div className="helpTitleStrong">How to choose settings</div>

			<ul className="helpList helpListXs helpListLoose">
				<li>
					<span className="helpInlineLabel">Base estimator</span> – AdaBoost works best with{' '}
					<span className="helpInlineStrong">weak learners</span>. The classic choice is a decision stump (a very
					shallow tree). If you use a strong learner (deep trees, complex models), AdaBoost can overfit quickly
					and become unstable.
					{!isReg && <p className="helpTextBodyXs">Tip: for classification, start with a shallow tree-like base learner.</p>}
				</li>

				<li>
					<span className="helpInlineLabel">Number of estimators</span> – number of boosting rounds (how many weak
					learners are added sequentially). Higher values can improve performance, but also increase training time
					and overfitting risk. Typical range: <span className="helpInlineStrong">50–500</span>. If you reduce the
					learning rate, you usually need more estimators.
				</li>

				<li>
					<span className="helpInlineLabel">Learning rate</span> – scales how much each new learner contributes.
					Smaller values make boosting more conservative and often improve generalization, but you typically need
					more estimators. Good starting points: <span className="helpInlineStrong">0.05–0.5</span>. If results are
					unstable, try lowering it.
				</li>

				{!isReg && (
					<li>
						<span className="helpInlineLabel">Algorithm</span> – controls the boosting variant. If you’re unsure, keep
						the default. Older sklearn versions used SAMME/SAMME.R; newer versions have changed defaults and may
						deprecate some options. Only change this if you know you need it.
					</li>
				)}

				<li>
					<span className="helpInlineLabel">Random state</span> – seed for reproducibility. Set to a fixed value
					(e.g. 42) to make results repeatable across runs (especially with stochastic base estimators).
				</li>

				<li>
					<span className="helpInlineLabel">Practical tuning recipe</span> – if you see overfitting: reduce the
					learning rate, reduce base estimator complexity, and/or add more data. If you see underfitting: increase
					estimators and/or allow slightly stronger base learners.
				</li>
			</ul>
		</div>
	);
}

function XGBoostDetailsText() {
	return (
		<div className="helpStack helpStackXs">
			<div className="helpTitleStrong">How to choose settings</div>

			<ul className="helpList helpListXs helpListLoose">
				<li>
					<span className="helpInlineLabel">Number of estimators</span> – number of boosted trees. More trees can
					improve performance, but increase training time and overfitting risk. Typical range:{' '}
					<span className="helpInlineStrong">200–2000</span> (depends heavily on learning rate and dataset size).
				</li>

				<li>
					<span className="helpInlineLabel">Learning rate</span> – how aggressively each tree updates the model.
					Smaller values are safer and often generalize better, but need more trees. Common starting points:{' '}
					<span className="helpInlineStrong">0.03–0.2</span>.
				</li>

				<li>
					<span className="helpInlineLabel">Max depth</span> – maximum depth of each tree. Deeper trees can capture more
					complex patterns but overfit more easily. Typical range: <span className="helpInlineStrong">3–10</span>.
					If you see overfitting, reduce this.
				</li>

				<li>
					<span className="helpInlineLabel">Subsample</span> – fraction of rows used to grow each tree. Values &lt; 1.0
					add randomness and often improve generalization (especially on noisy data). Try{' '}
					<span className="helpInlineStrong">0.6–0.9</span> as a starting range.
				</li>

				<li>
					<span className="helpInlineLabel">Column sample by tree</span> – fraction of features considered per tree.
					Reducing this (e.g. <span className="helpInlineStrong">0.5–0.9</span>) can reduce overfitting and help
					with high-dimensional inputs.
				</li>

				<li>
					<span className="helpInlineLabel">L2 regularization (lambda)</span> – larger values penalize large weights
					and can reduce overfitting. If your model overfits, try increasing it.
				</li>

				<li>
					<span className="helpInlineLabel">L1 regularization (alpha)</span> – encourages sparsity in leaf weights.
					Can help when many features are noisy or redundant.
				</li>

				<li>
					<span className="helpInlineLabel">Min child weight</span> – minimum “amount of information” needed in a leaf.
					Higher values make the algorithm more conservative (fewer, simpler splits), which can help reduce
					overfitting.
				</li>

				<li>
					<span className="helpInlineLabel">Gamma</span> – minimum loss reduction required to make a split. Higher values
					make splitting more conservative (often helps with overfitting).
				</li>

				<li>
					<span className="helpInlineLabel">Use early stopping (Advanced)</span> – enables an{' '}
					<span className="helpInlineStrong">internal validation split</span> from the training data, and stops adding
					trees once the validation metric stops improving. This is also what allows MENDer to show{' '}
					<span className="helpInlineStrong">learning curves</span> (training vs validation over boosting rounds).
					Turning this off is fine if you only care about the final held-out / k-fold metric, but then “best
					iteration/score” and learning curves may be unavailable.
				</li>

				<li>
					<span className="helpInlineLabel">Early stopping rounds (patience)</span> – how many rounds XGBoost will wait
					without improvement before stopping. Smaller values stop sooner (faster, less overfitting risk);
					larger values are more permissive. If left blank, MENDer chooses a reasonable default.
				</li>

				<li>
					<span className="helpInlineLabel">Eval set fraction (Advanced)</span> – fraction of the training fold
					reserved for internal early-stopping evaluation (typical range: <span className="helpInlineStrong">0.1–0.3</span>).
					Higher values give a more stable validation signal but leave fewer samples to fit the trees. This does{' '}
					<span className="helpInlineStrong">not</span> change your main train/test split (holdout/k-fold).
				</li>

				<li>
					<span className="helpInlineLabel">Internal eval metric vs final metric</span> – learning curves and “best
					score” are based on an internal training metric (e.g. logloss/mlogloss/rmse), chosen for stability.
					Your model card and confusion/ROC use your selected final metric (e.g. accuracy).
				</li>

				<li>
					<span className="helpInlineLabel">Number of jobs</span> – parallelism. Higher values use more CPU cores. If
					supported, <span className="helpInlineStrong">-1</span> means “use all cores”.
				</li>

				<li>
					<span className="helpInlineLabel">Random state</span> – seed for reproducibility. Set to a fixed value
					(e.g. 42) for repeatable runs.
				</li>

				<li>
					<span className="helpInlineLabel">Practical tuning recipe</span> – start with learning_rate 0.1, max_depth
					4–6, subsample/colsample 0.8, then tune depth/regularization to control overfitting. If underfitting,
					add trees or increase depth slightly.
				</li>
			</ul>

			<p className="helpTextBodyXs">
				Note: XGBoost requires the xgboost Python package to be installed in the backend environment.
			</p>
		</div>
	);
}

/* ---------- router ---------- */

export default function EnsembleHelpText({ kind, effectiveTask, votingType }) {
	if (kind === 'voting') {
		return <VotingDetailsText effectiveTask={effectiveTask} votingType={votingType} />;
	}
	if (kind === 'bagging') {
		return <BaggingDetailsText />;
	}
	if (kind === 'adaboost') {
		return <AdaBoostDetailsText effectiveTask={effectiveTask} />;
	}
	if (kind === 'xgboost') {
		return <XGBoostDetailsText />;
	}

	return <p className="helpTextBodyXs">Help text for this ensemble type is not available yet.</p>;
}

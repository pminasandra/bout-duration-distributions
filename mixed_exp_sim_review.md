## Discrepencies to fix
Smallish differences between text and code.

* p and p-1 are reversed between text and code
* number of simulations are different between text and code
* Equation in the text (appendix line 212) only partially specifies the distribution. In particular it uses exp to mean exponential distributuion of x.

## Philosophical Concerns

I don't think these simulations strengthen the arguments made in the paper. My understanding is that you are trying to address reviewers who are curious whether the distributions you have are composed of biologically meaningful memoryless component distributions. The answer which you address in the text is that you do not have any evidence that shows this is the case, nor do you have evidence that this is not the case. The only way biologically meaningful way you are able to decompose the distributions that you find is by individual and those sub distributions are not memoryless. There may or may not exists some other biologicially meaningful factor along which you could decompose your distributions into constituent memoryless distributions. In my opinion, this argument stands on its own, and the simulatuions run do not effectively bolster it.

There are a couple possible claims that seem to be relevant here.

**Claim 1** Observing powerlaw/truncated power law distributions in nature necessarily implies (or strongly suggest) that these distributions are composed of constintuent biologically meaningful memoryless processes. 

**Claim 2** All or most aggregations of memoryless processes in nature result in the appearance of a powerlaw/truncated power law distribution.

Both of these claims seem almost certainly false to me. At a minimum, whoever is making this claim would have the burden to support it. I don't think the Petrovskii paper is making either of these claims. I think they are claiming something much more limited. If they are claiming one of these things analytically and you are trying to disprove it, ideally you should show the flaw in their analysis.

Setting aside the analysis in the Patrovskii paper, Claim #1 seems more relevant to the interpreation of your data. However, I don't think this simluation addresses this claim at all. To really address it, I think you would want to propose an alternative mechanism that could generate your data.

Claim #2 seems less related to your paper, but more related to your simulation. If Claim #2 were true, it would support Claim #1, but even if Claim #2 were true, it wouldn't make Claim #1 necessarily true. Proving Claim #2 false likewise doesn't automatically prove Claim #1 false.

The only thing your simlation shows is that there exists some combinitions memoryless processes that don't result in powerlaw-like aggregates. Which... 

A) It should be possible to show this through example analytically. B) Only disproves the "strong" version of claim 2 ("All") C) Does little if anything to disprove claim #1.


To address these concerns, I think it would be helpful to identify 1) the specific claim your prospective reviewer may make (presumably based on the Petrovskii paper) that you are trying to disprove with this simulation (Is it one of the two I've listed or something else?) 2) if the claim is related to the Petrovskii paper, whether it is a claim that paper actually makes or a generalization that others made based on their finding 3) how your simlation relates to that claim...does it definitely disprove it or just offer evidence against?


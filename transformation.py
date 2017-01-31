ELECTRON_MASS = 0.00054857990924

class Transformation(object):
	def __init__(self,name,adduct_mass,charge = 1,multiplicity = 1,isotope_diff = 0,parent = None,vote = 1.0,adducts = "none",charge_name = None,isotope = "mono"):
		self.name = name
		self.adduct_mass = adduct_mass
		self.charge = charge
		self.multiplicity = multiplicity
		self.isotope_diff = isotope_diff
		self.parent = parent
		self.vote = vote
		self.adducts = adducts
		self.charge_name = charge_name
		self.isotope = isotope

	def transform(self,peak):
		M = peak.mass*abs(self.charge) + self.charge*ELECTRON_MASS - self.adduct_mass
		# M /= (1.0*self.multiplicity)
		# M -= self.isotope_diff
		M -= self.isotope_diff
		M /= (1.0*self.multiplicity)
		return M

	def reversetransform(self,M):
		# reverse transform
		p = M
		p *=(1.0*self.multiplicity)
		p += self.isotope_diff
		p = p + self.adduct_mass - self.charge*ELECTRON_MASS
		p = p/abs(self.charge)
		return p

	def __str__(self):
		return self.name

	def __repr__(self):
		return self.__str__()

def load_from_file(file_name,charge_counts = {},adduct_counts = {},isotope_counts = {},multiplicity_counts = {}):
	import yaml,re
	transformations = []
	all_vals = yaml.load(open(file_name,'r'))

	charge_probs = {}
	tot_ch = 0.0
	for c in all_vals['charge_carriers']:
		charge_probs[c] = all_vals['charge_carriers'][c]['v']
		charge_probs[c] += charge_counts.get(c,0)
		tot_ch += charge_probs[c]

	# print "Charge Probabilities"
	for c in charge_probs:
		charge_probs[c]/=tot_ch
		# print "{}: {:.3f}".format(c,charge_probs[c])

	adduct_probs = {}
	tot_ad = 0.0
	for a in all_vals['adducts']:
		adduct_probs[a] = all_vals['adducts'][a]['v']
		adduct_probs[a] += adduct_counts.get(a,0)
		tot_ad += adduct_probs[a]

	# print
	# print "Adduct Probabilities"
	for a in adduct_probs:
		adduct_probs[a] /= tot_ad
		# print "{}: {:.3f}".format(a,adduct_probs[a])


	iso_probs = {}
	tot_iso = 0.0
	for i in all_vals['isotopes']:
		iso_probs[i] = all_vals['isotopes'][i]['v']
		iso_probs[i] += isotope_counts.get(i,0)
		tot_iso += iso_probs[i]
	# print
	# print "Isotope Probabilities"
	for i in iso_probs:
		iso_probs[i] /= tot_iso
		# print "{}: {:.3f}".format(i,iso_probs[i])

	mult_probs = {}
	tot_mult = 0.0
	for m in all_vals['multiplicities']:
		mult_probs[m] = all_vals['multiplicities'][m]
		mult_probs[m] += multiplicity_counts.get(m,0)
		tot_mult += mult_probs[m]
	# print
	# print "Multiplicity Probabilities"
	for m in mult_probs:
		mult_probs[m] /= tot_mult
		# print "{}: {:.3f}".format(m,mult_probs[m])



	# Loop over adducts and fragments

	for c in all_vals['charge_carriers']:
		if 'm' in all_vals['charge_carriers'][c]:
			mult_vals = all_vals['charge_carriers'][c]['m']
		else:
			mult_vals = [1]

		for multiplicity in mult_vals:
			if multiplicity == 1:
				mult_str = ""
			else:
				mult_str = str(multiplicity)

			vote = mult_probs[str(multiplicity)]

			base_name =  "{}M{}".format(mult_str,c)
			# print name
			charge = 0
			charge_mass = 0
			vote *= charge_probs[c]
			# vote *= all_vals['charge_carriers'][c]['v']
			for a in all_vals['charge_carriers'][c]['g']:
				charge += all_vals['charges'][a]
				charge_mass += all_vals['masses'][a]

			# print "\tCharge: {}".format(charge)
			# transformations.append(Transformation(name,adduct_mass,charge=charge,multiplicity=multiplicity,vote=vote,charge_name= c))

			for ad in all_vals['adducts']:
				if not ad == 'none':
					name = "[{}M{}]{}".format(mult_str,ad,c)
				else:
					name = base_name
				# print name
				ad_vote = vote * adduct_probs[ad]
				if not ad == 'none':
					adduct_mass = charge_mass + all_vals['masses'][ad]
				else:
					adduct_mass = charge_mass

				final_vote = ad_vote * iso_probs['mono']
				t = Transformation(name,adduct_mass,charge=charge,multiplicity=multiplicity,vote=final_vote,adducts = ad,charge_name = c)
				transformations.append(t)

				# Note just the two isotopes -- should probably add more for multiplicity > 1
				newname = name + '[C13]'
				final_vote = ad_vote *iso_probs['C13']
				c13_diff = 1.0033548378
				c13 = Transformation(newname,t.adduct_mass,charge=t.charge,
					multiplicity = t.multiplicity,
					isotope_diff = c13_diff,
					parent = t,
					vote = final_vote,
					adducts = t.adducts,
					charge_name = t.charge_name,
					isotope = "C13")
				transformations.append(c13)

				newname = name + '[2C13]'
				final_vote = ad_vote *iso_probs['2C13']
				c13_diff = 1.0033548378
				c13_2 = Transformation(newname,t.adduct_mass,charge=t.charge,
					multiplicity = t.multiplicity,
					isotope_diff = 2*c13_diff,
					parent = c13,
					vote = final_vote,
					adducts = t.adducts,
					charge_name = t.charge_name,
					isotope = '2C13')
				transformations.append(c13_2)

	return transformations




	# return transformations
	# for tr in all_vals['transformations']:
	# 	multiplicity = all_vals['transformations'][tr]['n']
	# 	charge = 0
	# 	adduct_mass = 0
	# 	vote = all_vals['transformations'][tr]['v']
	# 	for a in all_vals['transformations'][tr]['g']:
	# 		if a in all_vals['charges']:
	# 			charge += all_vals['charges'][a]
	# 		if a in all_vals['masses']:
	# 			adduct_mass += all_vals['masses'][a]
	# 	adduct_list = all_vals['transformations'][tr]['g']
	# 	transformations.append(Transformation(tr,adduct_mass,charge=charge,multiplicity=multiplicity,vote=vote,adducts = adduct_list))

	# 	# for i in all_vals['isotopes']:
	# 	# 	isotope_diff = all_vals['masses'][i]
	# 	# 	newname = "{} [{}]".format(tr,i)
	# 	# 	transformations.append(Transformation(newname,adduct_mass,charge=charge,multiplicity=multiplicity,isotope_diff=isotope_diff))

	# 	if 'fragments' in all_vals:
	# 		for f in all_vals['fragments']:
	# 			fragment_mass = all_vals['masses'][f]
	# 			if multiplicity == 1:
	# 				newname = "[M-{}]".format(f)
	# 			else:
	# 				newname = "[2M-{}]".format(f)
	# 			for a in all_vals['transformations'][tr]['g']:
	# 				if a[0] == '-':
	# 					newname += a
	# 				else:
	# 					newname += '+' + a

	# 			frag_vote = all_vals['fragments'][f]['v']
	# 			transformations.append(Transformation(newname,adduct_mass,charge=charge,multiplicity=multiplicity,fragment_mass=fragment_mass,vote = vote*frag_vote,adducts = adduct_list,fragments = [f]))
	# 			# for i in all_vals['isotopes']:
	# 			# 	isotope_diff = all_vals['masses'][i]
	# 			# 	newname = "{} [{}]".format(newname,i)
	# 			# 	transformations.append(Transformation(newname,adduct_mass,charge=charge,
	# 			# 											multiplicity=multiplicity,
	# 			# 											isotope_diff=isotope_diff,
	# 			# 											fragment_mass = fragment_mass))


	# final_transformations = []
	# # if 'isotopes' in all_vals:
	# # 	for t in transformations:
	# # 		final_transformations.append(t)
	# # 		for i in all_vals['isotopes']:
	# # 			new_name = t.name + '[{}]'.format(i)
	# # 			isotope_diff = all_vals['masses'][i]
	# # 			iso_vote = all_vals['isotopes'][i]['v']
	# # 			final_transformations.append(Transformation(new_name,t.adduct_mass,charge=t.charge,
	# # 														multiplicity = t.multiplicity,
	# # 														isotope_diff = isotope_diff,
	# # 														fragment_mass = t.fragment_mass,
	# # 														parent = t,
	# # 														vote = t.vote*iso_vote,
	# # 														adducts = t.adducts,
	# # 														fragments = t.fragments))
	# for t in transformations:
	# 	final_transformations.append(t)
	# 	iso_vote = 0.9 # fairly random choice!
	# 	c13_diff = 1.0033548378
	# 	new_name = t.name + '[C13]'
	# 	c13 = Transformation(new_name,t.adduct_mass,charge=t.charge,
	# 		multiplicity = t.multiplicity,
	# 		isotope_diff = c13_diff,
	# 		parent = t,
	# 		vote = t.vote*iso_vote,
	# 		adducts = t.adducts,
	# 		charge_name = t.charge_name)
	# 	final_transformations.append(c13)
	# 	new_name = t.name + '[2C13]'
	# 	c13_2 = Transformation(new_name,t.adduct_mass,charge=t.charge,
	# 		multiplicity = t.multiplicity,
	# 		isotope_diff = 2*c13_diff,
	# 		parent = c13,
	# 		vote = c13.vote*iso_vote,
	# 		adducts = t.adducts,
	# 		charge_name = t.charge_name)

	# 	final_transformations.append(c13_2)

	# 	if t.multiplicity > 1:
	# 		# add two more!
	# 		new_name = t.name + '[3C13]'
	# 		c13_3 = Transformation(new_name,t.adduct_mass,charge=t.charge,
	# 		multiplicity = t.multiplicity,
	# 		isotope_diff = 3*c13_diff,
	# 		parent = c13_2,
	# 		vote = c13.vote*iso_vote,
	# 		adducts = t.adducts,
	# 		charge_name = t.charge_name)
	# 		final_transformations.append(c13_3)

	# 		new_name = t.name + '[4C13]'
	# 		c13_4 = Transformation(new_name,t.adduct_mass,charge=t.charge,
	# 		multiplicity = t.multiplicity,
	# 		isotope_diff = 4*c13_diff,
	# 		parent = c13_3,
	# 		vote = c13.vote*iso_vote,
	# 		adducts = t.adducts,
	# 		charge_name = t.charge_name)
	# 		final_transformations.append(c13_4)


	# else:
	# 	final_transformations = transformations

	# print transformations
	return final_transformations

class Counts(object):
	def __init__(self):
		self.charge_counts = {}
		self.adduct_counts = {}
		self.isotope_counts = {}
		self.multiplicity_counts = {}

	def update(self,transformation):
		if not transformation.charge_name in self.charge_counts:
			self.charge_counts[transformation.charge_name] = 1
		else:
			self.charge_counts[transformation.charge_name] += 1
		if not transformation.adducts in self.adduct_counts:
			self.adduct_counts[transformation.adducts] = 1
		else:
			self.adduct_counts[transformation.adducts] += 1
		if not transformation.isotope in self.isotope_counts:
			self.isotope_counts[transformation.isotope] = 1
		else:
			self.isotope_counts[transformation.isotope] += 1
		mu = str(transformation.multiplicity)
		if not mu in self.multiplicity_counts:
			self.multiplicity_counts[mu] = 1
		else:
			self.multiplicity_counts[mu] += 1

	def __str__(self):
		line = "\n\n"
		line += "Charge counts\n"
		for c in self.charge_counts:
			line += "{}: {}\n".format(c,self.charge_counts[c])
		line += "\nAdduct counts\n"
		for a in self.adduct_counts:
			line += "{}: {}\n".format(a,self.adduct_counts[a])
		line += "\nIsotope counts\n"
		for i in self.isotope_counts:
			line += "{}: {}\n".format(i,self.isotope_counts[i])
		line += "\nMultiplicity counts\n"
		for m in self.multiplicity_counts:
			line += "{}: {}\n".format(m,self.multiplicity_counts[m])
		return line

if __name__=='__main__':
	load_from_file('pos_transformations.yml')
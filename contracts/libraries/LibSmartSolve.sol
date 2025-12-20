// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { IDiamondCut } from "../interfaces/IDiamondCut.sol";

/**
 * @title LibSmartSolve
 * @notice Internal library providing diamond storage and upgrade logic for the SmartSolve system.
 *         Implements the EIP-2535 Diamond Storage pattern.
 */
library LibSmartSolve {
    // Fixed storage slot used for diamond storage (EIP-2535 pattern)
    bytes32 internal constant DIAMOND_STORAGE_POSITION =
        keccak256("smart-solve.diamond.storage");

    // ------------------------------------------------------------
    // Data Structures
    // ------------------------------------------------------------

    /**
    * @notice Holds info about one function selector
    * @param facetAddress The facet that implements this selector
    * @param selectorPosition The index of this selector inside that facet’s selector array
     */
    struct FacetAddressAndSelectorPosition {
        address facetAddress;
        uint16 selectorPosition; // index in facetFunctionSelectors
    }

    /**
    * @notice Holds all selectors for a given facet
    * @param selectors The array of function selectors for this facet
    * @param facetAddressPosition The index of this facet in the global facetAddresses array
     */ 
    struct FacetFunctionSelectors {
        bytes4[] selectors; // functions this facet implements
        uint16 facetAddressPosition; // index in facetAddresses array
    }

    /**
     * @notice Core storage layout for the SmartSolve Diamond (EIP-2535).
     *         Lives at the fixed DIAMOND_STORAGE_POSITION slot to avoid collisions.
     *
     *          selectorToFacetAndPosition:
     *                Maps each function selector to:
     *                    - the facet implementing it
     *                    - the selector’s index inside that facet’s selector array
     *
     *          facetFunctionSelectors:
     *                Maps a facet address to:
     *                    - the list of selectors it provides
     *                   - its index in the global facetAddresses list
     *
     *          facetAddresses:
     *                 List of all facet addresses currently linked to the diamond.
     *                 Enables enumeration of facets and introspection (Loupe).
     *
     *          contractOwner:
     *                 Single privileged owner used for upgrade authorization.
     *
     *          supportedInterfaces:
     *                 ERC-165 interface ID → supported flag.
     *                Used by facets implementing ERC-165 detection logic.
     */
    struct DiamondStorage {
        mapping(bytes4 => FacetAddressAndSelectorPosition) selectorToFacetAndPosition;
        mapping(address => FacetFunctionSelectors) facetFunctionSelectors;
        address[] facetAddresses;
        address contractOwner;
        mapping(bytes4 => bool) supportedInterfaces; // for ERC-165
    }

    // ------------------------------------------------------------
    // Access to Storage
    // ------------------------------------------------------------

    /**
     * @notice Returns the diamond’s storage struct at the fixed EIP-2535 storage slot.
     *         Uses inline assembly to bind 'DiamondStorage' to the constant slot
     *         defined by DIAMOND_STORAGE_POSITION.
     * @return ds Pointer to the diamond storage layout
     */
    function diamondStorage() internal pure returns (DiamondStorage storage ds) {
        bytes32 position = DIAMOND_STORAGE_POSITION;
        assembly {
            ds.slot := position
        }
    }

    // ------------------------------------------------------------
    // Ownership
    // ------------------------------------------------------------

    /**
     * @notice Sets the contract owner address.
     *         Updates the owner stored in diamond storage.
     * @param  newOwner  Address to assign as the new owner
     * @return prevOwner The previous owner address
     */
    function setContractOwner(address newOwner) internal returns (address prevOwner) {
        DiamondStorage storage ds = diamondStorage();
        prevOwner = ds.contractOwner;
        ds.contractOwner = newOwner;
        return prevOwner;
    }

    /**
     * @notice Retrieves the current contract owner.
     * @return Address of the owner stored in diamond storage
     */
    function contractOwner() internal view returns (address) {
        return diamondStorage().contractOwner;
    }

    /**
     * @notice Reverts unless msg.sender equals the stored contract owner.
     * @dev Used by facets to enforce privileged upgrade access.
     */
    function enforceIsContractOwner() internal view {
        require(msg.sender == diamondStorage().contractOwner, "LibSmartSolve: Must be contract owner");
    }

    // ------------------------------------------------------------
    // DiamondCut Logic
    // ------------------------------------------------------------

    /**
     * @notice Applies a diamond upgrade by adding, replacing, or removing selectors.
     *         Iterates over the FacetCut list, performs the corresponding actions,
     *         emits the DiamondCut event, then optionally performs an init delegatecall.
     *
     * @param _diamondCut Array of facet modifications (add/replace/remove)
     * @param _init Target contract for optional initialization call
     * @param _calldata Encoded call data used when executing the init delegatecall
     */
    function diamondCut(
        IDiamondCut.FacetCut[] memory _diamondCut,
        address _init,
        bytes memory _calldata
    ) internal {
        for (uint256 i; i < _diamondCut.length; i++) {
            IDiamondCut.FacetCutAction action = _diamondCut[i].action;
            if (action == IDiamondCut.FacetCutAction.Add) {
                addFunctions(_diamondCut[i].facetAddress, _diamondCut[i].functionSelectors);
            } else if (action == IDiamondCut.FacetCutAction.Replace) {
                replaceFunctions(_diamondCut[i].facetAddress, _diamondCut[i].functionSelectors);
            } else if (action == IDiamondCut.FacetCutAction.Remove) {
                removeFunctions(_diamondCut[i].facetAddress, _diamondCut[i].functionSelectors);
            } else {
                revert("LibSmartSolve: Incorrect FacetCutAction");
            }
        }
        emit IDiamondCut.DiamondCut(_diamondCut, _init, _calldata);
        initializeDiamondCut(_init, _calldata);
    }

    // ------------------------------------------------------------
    // Helpers for DiamondCut
    // ------------------------------------------------------------

    /**
     * @notice Adds a list of function selectors to a facet.
     *         Registers the facet if new, appends selectors, and updates the
     *         selector→facet mapping. Reverts if any selector already exists.
     *
     * @param _facet Facet address providing the implementations
     * @param _selectors Function selectors to add
     */
    function addFunctions(address _facet, bytes4[] memory _selectors) internal {
        require(_facet != address(0), "LibSmartSolve: facet address is zero");
        DiamondStorage storage ds = diamondStorage();

        // if facet is new, push it to facetAddresses
        if (ds.facetFunctionSelectors[_facet].selectors.length == 0) {
            ds.facetFunctionSelectors[_facet].facetAddressPosition = uint16(ds.facetAddresses.length);
            ds.facetAddresses.push(_facet);
        }

        for (uint256 i; i < _selectors.length; i++) {
            bytes4 selector = _selectors[i];
            require(ds.selectorToFacetAndPosition[selector].facetAddress == address(0),
                "LibSmartSolve: selector already exists");
            ds.facetFunctionSelectors[_facet].selectors.push(selector);
            ds.selectorToFacetAndPosition[selector] =
                FacetAddressAndSelectorPosition(_facet, uint16(ds.facetFunctionSelectors[_facet].selectors.length - 1));
        }
    }

    /**
     * @notice Replaces existing selectors with implementations from a new facet.
     *         Each selector is removed, then re-added pointing to the new facet.
     *         Reverts if the new facet address is zero.
     *
     * @param _facet Facet supplying replacement implementations
     * @param _selectors Function selectors to replace
     */
    function replaceFunctions(address _facet, bytes4[] memory _selectors) internal {
        require(_facet != address(0), "LibSmartSolve: facet address is zero");
        for (uint256 i; i < _selectors.length; i++) {
            removeFunction(_selectors[i]);
            addFunctions(_facet, toSingletonArray(_selectors[i]));
        }
    }

    /**
     * @notice Removes a list of function selectors from the diamond.
     *         Deletes each selector mapping and removes empty facets.
     *
     * @param _selectors Function selectors to remove
     */
    function removeFunctions(address /*_facet*/, bytes4[] memory _selectors) internal {
        // _facet param is unused but kept for event consistency
        for (uint256 i; i < _selectors.length; i++) {
            removeFunction(_selectors[i]);
        }
    }

    /**
     * @notice Removes a single selector from the diamond.
     *         Performs in-place swap & pop on the facet’s selector array,
     *         updates selector positions, and removes facets that become empty.
     *
     * @param _selector The selector to remove
     */
    function removeFunction(bytes4 _selector) private {
        DiamondStorage storage ds = diamondStorage();
        FacetAddressAndSelectorPosition memory old = ds.selectorToFacetAndPosition[_selector];
        require(old.facetAddress != address(0), "LibSmartSolve: selector does not exist");

        // Get facet selectors array
        bytes4[] storage selectors = ds.facetFunctionSelectors[old.facetAddress].selectors;
        uint256 lastPos = selectors.length - 1;
        bytes4 lastSelector = selectors[lastPos];

        // Swap last selector into removed selector's position and pop the array
        selectors[old.selectorPosition] = lastSelector;
        ds.selectorToFacetAndPosition[lastSelector].selectorPosition = old.selectorPosition;
        selectors.pop();

        // If facet now empty, remove facet address
        if (selectors.length == 0) {
            uint16 lastAddrPos = uint16(ds.facetAddresses.length - 1);
            address lastAddr = ds.facetAddresses[lastAddrPos];
            ds.facetAddresses[ds.facetFunctionSelectors[old.facetAddress].facetAddressPosition] = lastAddr;
            ds.facetFunctionSelectors[lastAddr].facetAddressPosition =
                ds.facetFunctionSelectors[old.facetAddress].facetAddressPosition;
            ds.facetAddresses.pop();
            delete ds.facetFunctionSelectors[old.facetAddress];
        }

        delete ds.selectorToFacetAndPosition[_selector];
    }

    // ------------------------------------------------------------
    // Init Call
    // ------------------------------------------------------------

    /**
     * @notice Executes the optional initialization delegatecall after a diamond cut.
     *         Reverts if '_init' is zero but calldata is non-empty. Propagates
     *         revert data from failed delegatecalls.
     *
     * @param _init Target contract to execute initialization logic
     * @param _calldata Encoded call for the delegatecall
     */
    function initializeDiamondCut(address _init, bytes memory _calldata) private {
        if (_init == address(0)) {
            require(_calldata.length == 0, "LibSmartSolve: _init is zero but calldata is not empty");
        } else {
            (bool success, bytes memory error) = _init.delegatecall(_calldata);
            if (!success) {
                if (error.length > 0) {
                    // bubble up error
                    assembly {
                        revert(add(error, 32), mload(error))
                    }
                } else {
                    revert("LibSmartSolve: _init function reverted");
                }
            }
        }
    }

    // ------------------------------------------------------------
    // Utility
    // ------------------------------------------------------------
    
    /**
     * @notice Wraps a single bytes4 selector into a dynamic array.
     *         Used when replaceFunctions() needs to add a single selector.
     * @param selector Function selector to pack
     * @return arr Array containing exactly one selector
     */
    function toSingletonArray(bytes4 selector) private pure returns (bytes4[] memory arr) {
        arr = new bytes4[](1);
        arr[0] = selector;
    }
}